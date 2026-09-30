#!/usr/bin/env python3
"""
Vérification visuelle du dataset YOLO : dessine les boîtes annotées sur les tuiles.

Usage :
    python check_dataset.py --dataset dataset

Produit :
    dataset/_verif/*.jpg          une image par tuile positive, boîtes tracées
    dataset/_verif/planche.jpg    planche-contact des 24 premières tuiles positives
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

COULEURS = {0: (0, 0, 255), 1: (0, 200, 0)}  # BGR : rouge = passage, vert = mobilier


def lire_labels(txt):
    boites = []
    for ligne in txt.read_text().splitlines():
        p = ligne.split()
        if len(p) == 5:
            boites.append((int(p[0]), *map(float, p[1:])))
    return boites


def main(dataset, max_planche=24):
    dataset = Path(dataset)
    out = dataset / "_verif"
    out.mkdir(exist_ok=True)

    positives = []
    for split in ("train", "val"):
        n_tuiles = n_boites = 0
        for txt in sorted((dataset / "labels" / split).glob("*.txt")):
            boites = lire_labels(txt)
            if not boites:
                continue
            img = cv2.imread(str(dataset / "images" / split / txt.with_suffix(".jpg").name))
            if img is None:
                continue
            h, w = img.shape[:2]
            for cls, xc, yc, bw, bh in boites:
                x1, y1 = int((xc - bw / 2) * w), int((yc - bh / 2) * h)
                x2, y2 = int((xc + bw / 2) * w), int((yc + bh / 2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), COULEURS.get(cls, (255, 0, 0)), 3)
            dest = out / f"{split}_{txt.stem}.jpg"
            cv2.imwrite(str(dest), img)
            positives.append(img)
            n_tuiles += 1
            n_boites += len(boites)
        print(f"{split:5s} : {n_tuiles} tuile(s) positive(s), {n_boites} boîte(s)")

    if positives:
        vignettes = [cv2.resize(i, (320, 320)) for i in positives[:max_planche]]
        while len(vignettes) % 6:
            vignettes.append(np.zeros((320, 320, 3), np.uint8))
        lignes = [np.hstack(vignettes[i:i + 6]) for i in range(0, len(vignettes), 6)]
        cv2.imwrite(str(out / "planche.jpg"), np.vstack(lignes))
        print(f"Planche-contact : {out / 'planche.jpg'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset")
    main(ap.parse_args().dataset)
