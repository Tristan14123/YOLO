#!/usr/bin/env python3
"""
Détection de passages piétons et mobilier urbain via YOLOv8 (Ultralytics)
==========================================================================

Stack open source : ultralytics (AGPL-3.0), rasterio, shapely, geopandas.

Une orthophoto THR (5 cm) fait souvent plusieurs Go et des dizaines de
milliers de pixels de côté : on ne peut pas la passer telle quelle à YOLO.
Ce script la découpe en tuiles, lance l'inférence sur chaque tuile, puis
reprojette les boîtes détectées en coordonnées terrain et fusionne les
doublons de bord de tuile.

Deux usages :
  1. Inférence avec un modèle déjà entraîné :
     python yolo_detection.py infer --ortho ortho.tif --weights best.pt --out detections.gpkg

  2. Entraînement d'un modèle sur tes propres annotations :
     python yolo_detection.py train --data data.yaml --epochs 100

IMPORTANT : un modèle YOLO pré-entraîné COCO ne connaît PAS les classes
"passage piéton" / "mobilier urbain". Sans poids déjà spécialisés, il faut
annoter un jeu d'exemples (voir section ENTRAÎNEMENT en bas de fichier)
avant que l'inférence ne produise des résultats exploitables.
"""

import argparse
from pathlib import Path

import numpy as np
import pyproj
import rasterio
from rasterio.windows import Window
from shapely.geometry import box
import geopandas as gpd
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# 1. DÉCOUPAGE EN TUILES + INFÉRENCE GÉORÉFÉRENCÉE
# ---------------------------------------------------------------------------

def iter_tiles(src, tile_size=1024, overlap=128):
    """Génère des fenêtres de lecture qui se chevauchent légèrement,
    pour éviter de couper un objet pile sur une frontière de tuile."""
    width, height = src.width, src.height
    step = tile_size - overlap
    for top in range(0, height, step):
        for left in range(0, width, step):
            w = min(tile_size, width - left)
            h = min(tile_size, height - top)
            if w <= 0 or h <= 0:
                continue
            yield Window(left, top, w, h)


def infer_on_orthophoto(ortho_path, weights_path, out_gpkg,
                          tile_size=640, overlap=64,
                          conf_threshold=0.35, iou_threshold=0.5, force_epsg=None):
    model = YOLO(weights_path)
    device = pick_device()
    print(f"Inférence sur : {device}")
    class_names = model.names  # dict {id: nom_classe}, défini par l'entraînement

    all_records = []

    with rasterio.open(ortho_path) as src:
        if src.crs is not None:
            crs = pyproj.CRS.from_user_input(src.crs.to_wkt())
        elif force_epsg is not None:
            crs = pyproj.CRS.from_epsg(force_epsg)
        else:
            raise SystemExit("Orthophoto sans CRS embarqué : relance avec --epsg 2154.")
        transform = src.transform

        tiles = list(iter_tiles(src, tile_size, overlap))
        print(f"Orthophoto découpée en {len(tiles)} tuiles de {tile_size}px "
              f"(chevauchement {overlap}px)")

        for i, window in enumerate(tiles):
            img = src.read([1, 2, 3], window=window)
            img = np.transpose(img, (1, 2, 0))  # (H, W, 3) pour ultralytics

            results = model.predict(
                img, conf=conf_threshold, iou=iou_threshold, verbose=False, device=device
            )[0]

            if results.boxes is None or len(results.boxes) == 0:
                continue

            # Transform propre à cette tuile (décalage de la fenêtre)
            tile_transform = src.window_transform(window)

            for b in results.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                cls_id = int(b.cls[0])
                conf = float(b.conf[0])

                # Coin haut-gauche et bas-droit en coordonnées terrain
                gx1, gy1 = tile_transform * (x1, y1)
                gx2, gy2 = tile_transform * (x2, y2)

                geom = box(min(gx1, gx2), min(gy1, gy2), max(gx1, gx2), max(gy1, gy2))
                all_records.append({
                    "geometry": geom,
                    "classe": class_names.get(cls_id, str(cls_id)),
                    "confiance": conf,
                    "tile_id": i,
                })

            if (i + 1) % 50 == 0:
                print(f"  ... {i + 1}/{len(tiles)} tuiles traitées, "
                      f"{len(all_records)} détections cumulées")

    if not all_records:
        print("Aucune détection sur l'ensemble de l'orthophoto.")
        return gpd.GeoDataFrame(geometry=[], crs=crs)

    gdf = gpd.GeoDataFrame(all_records, crs=crs)
    gdf = deduplicate_overlaps(gdf)

    for classe in gdf["classe"].unique():
        sub = gdf[gdf["classe"] == classe]
        layer_name = classe.replace(" ", "_")
        sub.to_file(out_gpkg, layer=layer_name, driver="GPKG")
        print(f"  -> {len(sub)} objet(s) '{classe}' exporté(s) dans la couche '{layer_name}'")

    return gdf


def deduplicate_overlaps(gdf, iou_threshold=0.5):
    """Fusionne les détections dupliquées dans les zones de chevauchement
    entre tuiles adjacentes (NMS géométrique simple, par classe)."""
    keep_all = []
    for classe in gdf["classe"].unique():
        sub = gdf[gdf["classe"] == classe].sort_values("confiance", ascending=False)
        kept = []
        for idx, row in sub.iterrows():
            geom = row.geometry
            is_dup = False
            for k in kept:
                inter = geom.intersection(k.geometry).area
                union = geom.union(k.geometry).area
                if union > 0 and inter / union > iou_threshold:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(row)
        keep_all.extend(kept)
    return gpd.GeoDataFrame(keep_all, crs=gdf.crs)


# ---------------------------------------------------------------------------
# 2. COMPARAISON MULTI-MILLÉSIMES (créations / suppressions) — inchangé
# ---------------------------------------------------------------------------

def compare_epochs(gdf_ancien, gdf_recent, distance_tolerance=1.5):
    ancien_buff = gdf_ancien.copy()
    ancien_buff["geometry"] = ancien_buff.geometry.buffer(distance_tolerance)

    recent_buff = gdf_recent.copy()
    recent_buff["geometry"] = recent_buff.geometry.buffer(distance_tolerance)

    joined_new = gpd.sjoin(gdf_recent, ancien_buff, how="left", predicate="within")
    crees = gdf_recent[joined_new["index_right"].isna()]

    joined_old = gpd.sjoin(gdf_ancien, recent_buff, how="left", predicate="within")
    supprimes = gdf_ancien[joined_old["index_right"].isna()]

    inchanges = gdf_recent[~joined_new["index_right"].isna()]
    return crees, supprimes, inchanges


# ---------------------------------------------------------------------------
# 3. ENTRAÎNEMENT D'UN MODÈLE CUSTOM
# ---------------------------------------------------------------------------
# Pré-requis : un jeu d'images annotées au format YOLO (un .txt par image,
# une ligne par objet : classe x_centre y_centre largeur hauteur, normalisés
# 0-1). L'annotation peut se faire dans QGIS (plugin "Deep Learning Tools"
# ou export de digitalisation manuelle) ou avec un outil dédié type
# LabelImg / CVAT / Roboflow (tous open source ou gratuits en usage local).
#
# Structure attendue :
#   dataset/
#     images/train/*.jpg
#     images/val/*.jpg
#     labels/train/*.txt
#     labels/val/*.txt
#
# Fichier data.yaml attendu (exemple, à créer à côté du dataset) :
#
#   path: /chemin/vers/dataset
#   train: images/train
#   val: images/val
#   names:
#     0: passage_pieton
#     1: mobilier_urbain

def pick_device():
    """cuda (Windows/Linux + NVIDIA), mps (Mac Apple Silicon) ou cpu. Ultralytics
    ne choisit pas mps tout seul sur Mac : on le force ici."""
    import torch
    if torch.cuda.is_available():
        return 0
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_model(data_yaml, epochs=100, imgsz=1024, base_model="yolo26n.pt",
                device=None, batch=16):
    model = YOLO(base_model)  # part d'un modèle pré-entraîné (transfer learning)
    device = device or pick_device()
    print(f"Entraînement sur : {device}")
    # flipud=0.5 : en vue aérienne, "haut" et "bas" n'ont pas de sens -> augmentation sûre
    model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, device=device,
                batch=batch, flipud=0.5, workers=0 if device == "mps" else 8)
    print("Entraînement terminé. Les poids se trouvent dans runs/detect/train*/weights/best.pt")


# ---------------------------------------------------------------------------
# 4. CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    p_infer = sub.add_parser("infer", help="Lancer la détection sur une orthophoto")
    p_infer.add_argument("--ortho", required=True, help="Orthophoto GeoTIFF (5cm)")
    p_infer.add_argument("--weights", required=True, help="Poids YOLO entraînés (.pt)")
    p_infer.add_argument("--out", default="detections.gpkg")
    p_infer.add_argument("--tile-size", type=int, default=640)
    p_infer.add_argument("--overlap", type=int, default=64)
    p_infer.add_argument("--conf", type=float, default=0.35)
    p_infer.add_argument("--epsg", type=int, default=2154, help="CRS forcé si absent du fichier")

    p_train = sub.add_parser("train", help="Entraîner un modèle YOLO custom")
    p_train.add_argument("--data", required=True, help="Fichier data.yaml")
    p_train.add_argument("--epochs", type=int, default=100)
    p_train.add_argument("--imgsz", type=int, default=640)
    p_train.add_argument("--base-model", default="yolo26n.pt")
    p_train.add_argument("--device", default=None, help="cpu, mps, 0... (défaut : auto)")
    p_train.add_argument("--batch", type=int, default=16)

    args = parser.parse_args()

    if args.mode == "infer":
        infer_on_orthophoto(
            args.ortho, args.weights, args.out,
            tile_size=args.tile_size, overlap=args.overlap, conf_threshold=args.conf,
            force_epsg=args.epsg
        )
    elif args.mode == "train":
        train_model(args.data, epochs=args.epochs, imgsz=args.imgsz, base_model=args.base_model,
                    device=args.device, batch=args.batch)