#!/usr/bin/env python3
"""
Préparation d'un dataset YOLO à partir d'orthophotos PCRS + digitalisation QGIS
================================================================================

--ortho accepte : un GeoTIFF, un VRT (mosaïque virtuelle) ou un DOSSIER de dalles.

Ce qui change par rapport à un découpage naïf :
  * seules les dalles qui recoupent tes annotations sont lues ;
  * on garde toutes les tuiles positives + un échantillon de tuiles vides
    (--neg-ratio par tuile positive) au lieu des ~95 % de tuiles vides ;
  * le split train/val se fait par GROUPE spatial (dalle entière si tu en as
    plusieurs, sinon blocs de --block x --block tuiles) pour éviter que deux
    tuiles voisines d'un même passage tombent l'une en train, l'autre en val ;
  * val contient toujours des tuiles positives (objectif : --val-ratio des positifs).

IMPORTANT : dans chaque dalle utilisée, annote TOUS les passages piétons visibles.
Les tuiles "vides" servent d'exemples négatifs : un passage non annoté y serait
appris comme "ce n'est pas un passage".

Usage :
    python prepare_dataset.py build --ortho pcrs_tout.vrt --annotations passage.gpkg --out dataset
    python prepare_dataset.py build --ortho pcrs/ --annotations passage.gpkg --out dataset --force

Sortie (prête pour yolo_detection.py train --data dataset/data.yaml) :
    dataset/images/{train,val}/*.jpg   dataset/labels/{train,val}/*.txt
    dataset/data.yaml                  dataset/manifest.csv
"""

import argparse
import csv
import math
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import pyproj
import rasterio
from rasterio.transform import array_bounds
from rasterio.windows import Window
from shapely.geometry import box

CLASSES = {"passage_pieton": 0, "mobilier_urbain": 1}
DEFAULT_EPSG = 2154
POINT_BOX_RADIUS_M = 0.4   # rayon de la boîte autour d'un point de mobilier urbain
MIN_VISIBLE = 0.3          # fraction minimale d'un objet visible dans la tuile pour l'annoter


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def list_dalles(path):
    p = Path(path)
    if p.is_dir():
        dalles = sorted(f for f in p.iterdir() if f.suffix.lower() in (".tif", ".tiff", ".vrt"))
        if not dalles:
            sys.exit(f"Aucun .tif/.vrt trouvé dans {p}")
        return dalles
    if not p.exists():
        sys.exit(f"Fichier introuvable : {p}")
    return [p]


def get_crs(src, force_epsg):
    if src.crs is not None:
        return pyproj.CRS.from_user_input(src.crs.to_wkt())
    if force_epsg is None:
        sys.exit("Orthophoto sans CRS embarqué : relance avec --epsg 2154 (ou ton code EPSG).")
    return pyproj.CRS.from_epsg(force_epsg)


def window_starts(size, tile):
    """Départs de fenêtres couvrant toute la dalle ; la dernière est recalée sur le bord."""
    if size <= tile:
        return [0]
    starts = list(range(0, size - tile + 1, tile))
    if starts[-1] + tile < size:
        starts.append(size - tile)
    return starts


def to_yolo(geom, tr, w, h):
    """Géométrie terrain -> (boîte YOLO normalisée, fraction visible) dans la tuile, ou None."""
    inv = ~tr
    if geom.geom_type == "Point":
        r = POINT_BOX_RADIUS_M
        minx, miny, maxx, maxy = geom.x - r, geom.y - r, geom.x + r, geom.y + r
    else:
        minx, miny, maxx, maxy = geom.bounds
    xa, ya = inv * (minx, maxy)
    xb, yb = inv * (maxx, miny)
    x0, x1 = sorted((xa, xb))
    y0, y1 = sorted((ya, yb))
    full = (x1 - x0) * (y1 - y0)
    cx0, cx1, cy0, cy1 = max(0, x0), min(w, x1), max(0, y0), min(h, y1)
    if full <= 0 or cx1 <= cx0 or cy1 <= cy0:
        return None
    frac = (cx1 - cx0) * (cy1 - cy0) / full
    return ((cx0 + cx1) / 2 / w, (cy0 + cy1) / 2 / h, (cx1 - cx0) / w, (cy1 - cy0) / h), frac


# ---------------------------------------------------------------------------
# Phase 1 : planification (aucune image lue, très léger)
# ---------------------------------------------------------------------------

def plan_dalle(path, ann, tile_size, epsg):
    with rasterio.open(path) as src:
        crs = get_crs(src, epsg)
        a = ann if ann.crs == crs else ann.to_crs(crs)
        b = src.bounds
        if len(a.sindex.query(box(b.left, b.bottom, b.right, b.top), predicate="intersects")) == 0:
            return None

        H, W = src.height, src.width
        w, h = min(tile_size, W), min(tile_size, H)
        stem = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(path).stem)
        tiles = []
        for r, top in enumerate(window_starts(H, tile_size)):
            for c, left in enumerate(window_starts(W, tile_size)):
                win = Window(left, top, w, h)
                tr = src.window_transform(win)
                bounds = array_bounds(h, w, tr)  # (west, south, east, north)
                idx = a.sindex.query(box(*bounds), predicate="intersects")
                labels, ambiguous = [], False
                for i in idx:
                    row = a.iloc[i]
                    res = to_yolo(row.geometry, tr, w, h)
                    if res is None:
                        continue
                    (xc, yc, bw, bh), frac = res
                    if frac >= MIN_VISIBLE:
                        labels.append(f"{CLASSES[row['classe']]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                    else:
                        ambiguous = True  # objet à peine visible : tuile inutilisable comme négatif
                tiles.append({"path": Path(path), "stem": stem, "r": r, "c": c,
                              "win": (left, top, w, h), "labels": labels, "ambiguous": ambiguous,
                              "name": f"{stem}_r{r:03d}_c{c:03d}"})
        return tiles


# ---------------------------------------------------------------------------
# Phase 2 : sélection, split, écriture
# ---------------------------------------------------------------------------

def assign_splits(selected, n_pos_total, val_ratio, use_dalle_groups, block, rng):
    for t in selected:
        t["group"] = t["stem"] if use_dalle_groups else f'{t["stem"]}_{t["r"] // block}_{t["c"] // block}'
    groups = defaultdict(list)
    for t in selected:
        groups[t["group"]].append(t)
    pos_groups = [g for g, ts in groups.items() if any(x["labels"] for x in ts)]

    if len(pos_groups) < 2:
        print("ATTENTION : moins de 2 groupes spatiaux avec des passages -> split aléatoire par tuile "
              "(la validation sera moins fiable). Annote sur plus de zones/dalles.")
        rng.shuffle(selected)
        n_val = max(1, int(len(selected) * val_ratio))
        for i, t in enumerate(selected):
            t["split"] = "val" if i < n_val else "train"
        return

    order = list(groups)
    rng.shuffle(order)
    target = math.ceil(val_ratio * n_pos_total)
    val_pos, val_groups = 0, set()
    for g in order:
        gp = sum(1 for x in groups[g] if x["labels"])
        if gp > 0 and val_pos < target:
            val_groups.add(g)
            val_pos += gp
        elif gp == 0 and rng.random() < val_ratio:
            val_groups.add(g)
    for t in selected:
        t["split"] = "val" if t["group"] in val_groups else "train"


def write_dataset(selected, out):
    stats = defaultdict(lambda: [0, 0, 0])  # split -> [positives, négatives, boîtes]
    skipped = 0
    opened = {}
    manifest = []
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    for t in selected:
        src = opened.get(t["path"]) or opened.setdefault(t["path"], rasterio.open(t["path"]))
        left, top, w, h = t["win"]
        img = np.transpose(src.read([1, 2, 3], window=Window(left, top, w, h)), (1, 2, 0))
        if (img.sum(axis=2) == 0).mean() > 0.5:  # zone hors emprise (nodata)
            skipped += 1
            continue
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        (out / "images" / t["split"] / f'{t["name"]}.jpg').write_bytes(buf.tobytes())
        (out / "labels" / t["split"] / f'{t["name"]}.txt').write_text("\n".join(t["labels"]))
        s = stats[t["split"]]
        s[0 if t["labels"] else 1] += 1
        s[2] += len(t["labels"])
        manifest.append([t["name"], t["split"], t["path"].name, int(bool(t["labels"])), len(t["labels"])])
    for src in opened.values():
        src.close()

    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["tuile", "split", "dalle", "positive", "nb_boites"])
        wr.writerows(manifest)

    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n"
        + "\n".join(f"  {v}: {k}" for k, v in CLASSES.items()) + "\n")

    for split in ("train", "val"):
        p, n, b = stats[split]
        print(f"{split:5s} : {p} tuile(s) positive(s) ({b} boîtes) + {n} tuile(s) vide(s)")
    if skipped:
        print(f"{skipped} tuile(s) ignorée(s) car hors emprise (nodata)")


def cmd_build(args):
    out = Path(args.out)
    for d in (out / "images", out / "labels"):
        if d.exists() and any(d.iterdir()):
            if not args.force:
                sys.exit(f"{d} n'est pas vide. Relance avec --force pour repartir de zéro.")
            shutil.rmtree(d)

    dalles = list_dalles(args.ortho)
    ann = gpd.read_file(args.annotations)
    if "classe" not in ann.columns:
        sys.exit("La couche d'annotations doit avoir un champ 'classe'.")
    inconnues = set(ann["classe"].dropna().unique()) - set(CLASSES)
    if inconnues:
        print(f"Classes ignorées (inconnues) : {sorted(inconnues)}")
    ann = ann[ann["classe"].isin(CLASSES)].reset_index(drop=True)
    if ann.crs is None:
        with rasterio.open(dalles[0]) as src:
            ann = ann.set_crs(get_crs(src, args.epsg))
        print("Aucun CRS sur le GeoPackage : supposé identique à celui de l'orthophoto.")
    print(f"{len(ann)} annotation(s) valides ; {len(dalles)} fichier(s) ortho à examiner")

    tiles = []
    for d in dalles:
        plan = plan_dalle(d, ann, args.tile_size, args.epsg)
        if plan is None:
            print(f"  {d.name} : aucune annotation -> ignorée")
            continue
        n_pos = sum(1 for t in plan if t["labels"])
        print(f"  {d.name} : {n_pos} tuile(s) positive(s) sur {len(plan)}")
        tiles += plan

    positives = [t for t in tiles if t["labels"]]
    negatives = [t for t in tiles if not t["labels"] and not t["ambiguous"]]
    if not positives:
        sys.exit("Aucune tuile positive : les annotations ne recouvrent aucune des orthophotos fournies.")

    rng = random.Random(args.seed)
    n_neg = min(len(negatives), round(args.neg_ratio * len(positives)))
    selected = positives + rng.sample(negatives, n_neg)
    print(f"Sélection : {len(positives)} positives + {n_neg} vides (sur {len(negatives)} disponibles)")

    dalles_pos = {t["stem"] for t in positives}
    use_dalle_groups = len(dalles_pos) >= 2
    print("Split par " + ("dalle entière" if use_dalle_groups
                          else f"blocs de {args.block}x{args.block} tuiles"))
    assign_splits(selected, len(positives), args.val_ratio, use_dalle_groups, args.block, rng)
    write_dataset(selected, out)
    print(f"data.yaml : {out / 'data.yaml'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    b = sub.add_parser("build", help="Construire le dataset YOLO")
    b.add_argument("--ortho", required=True, help="GeoTIFF, VRT ou dossier de dalles")
    b.add_argument("--annotations", required=True, help="GeoPackage QGIS (champ 'classe')")
    b.add_argument("--out", required=True)
    b.add_argument("--tile-size", type=int, default=640)
    b.add_argument("--val-ratio", type=float, default=0.2)
    b.add_argument("--neg-ratio", type=float, default=1.0, help="tuiles vides par tuile positive")
    b.add_argument("--block", type=int, default=4, help="taille des blocs de split (en tuiles)")
    b.add_argument("--epsg", type=int, default=DEFAULT_EPSG, help="CRS forcé si absent du fichier")
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--force", action="store_true", help="vide images/ et labels/ existants")
    b.set_defaults(func=cmd_build)
    args = ap.parse_args()
    args.func(args)