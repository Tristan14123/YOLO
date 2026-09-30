# Projet de Détection de Passages Piétons avec YOLO

Ce projet utilise l'intelligence artificielle (YOLOv8) pour détecter automatiquement des passages piétons et du mobilier urbain sur des orthophotos (photos aériennes haute résolution).

## 📚 Vue d'ensemble pour débutants

### Qu'est-ce que YOLO ?
**YOLO** (You Only Look Once) est un algorithme de détection d'objets très populaire. Contrairement aux anciennes méthodes qui analysent une image plusieurs fois, YOLO ne regarde l'image qu'une seule fois pour détecter tous les objets simultanément. C'est ce qui le rend très rapide !

### Pourquoi découper les orthophotos ?
Les orthophotos du PCRS (Très Haute Résolution, 5cm/pixel) sont des fichiers gigantesques (plusieurs Go) avec des milliers de pixels de côté. YOLO ne peut pas traiter une image aussi grande en une seule fois. Il faut donc :
1. **Découper** l'orthophoto en petits morceaux (tuiles) de 640x640 pixels
2. **Analyser** chaque tuile avec YOLO
3. **Recomposer** les résultats en coordonnées géographiques réelles

---

## 🛠️ Les Scripts Python

### 1. `prepa_dataset.py` - Préparation du Dataset

**But :** Préparer les données d'entraînement pour YOLO à partir d'orthophotos et d'annotations faites dans QGIS.

#### Qu'est-ce qu'un dataset d'entraînement ?
Pour qu'une IA apprenne à reconnaître des passages piétons, il faut lui montrer des exemples. Un dataset contient :
- **Images** : des morceaux d'orthophoto (tuiles)
- **Labels** : des fichiers texte qui indiquent où se trouvent les passages piétons sur chaque image
- **Split train/val** : 
  - `train` (80%) : pour enseigner à l'IA
  - `val` (20%) : pour tester si l'IA a bien appris

#### Workflow complet :

##### Étape 1 : Digitalisation dans QGIS
1. Charger l'orthophoto dans QGIS
2. Créer une couche vecteur (GeoPackage) avec un champ `classe`
3. Dessiner manuellement :
   - **passage_pieton** → comme un POLYGONE autour du passage
   - **mobilier_urbain** → comme un POINT sur l'objet

##### Étape 2 : Construction du dataset
```bash
python prepa_dataset.py build --ortho pcrs_tout.vrt --annotations passage.gpkg --out dataset
```

**Ce que fait le script :**
1. **Liste les dalles** : Il accepte un GeoTIFF, un VRT (mosaïque virtuelle) ou un dossier de dalles
2. **Planifie intelligemment** : Il ne lit que les dalles qui contiennent des annotations (gain de temps !)
3. **Découpe en tuiles** : Il découpe chaque dalle en tuiles de 640x640 pixels
4. **Convertit les annotations** : Il transforme les coordonnées géographiques QGIS en coordonnées pixels YOLO
5. **Sélectionne les tuiles utiles** :
   - Garde **toutes** les tuiles avec des passages piétons (positives)
   - Ajoute un **échantillon** de tuiles vides (négatives) pour apprendre à ne pas détecter partout
6. **Split spatial intelligent** : Il évite que deux tuiles voisines du même passage se retrouvent l'une en train, l'autre en validation
7. **Crée les fichiers** :
   - `dataset/images/train/*.jpg` - Images d'entraînement
   - `dataset/images/val/*.jpg` - Images de validation
   - `dataset/labels/train/*.txt` - Labels d'entraînement
   - `dataset/labels/val/*.txt` - Labels de validation
   - `dataset/data.yaml` - Fichier de configuration YOLO
   - `dataset/manifest.csv` - Inventaire de toutes les tuiles

**⚠️ Important :** Dans chaque dalle utilisée, annote TOUS les passages piétons visibles. Les tuiles "vides" servent d'exemples négatifs - un passage non annoté serait appris comme "ce n'est pas un passage" !

#### Arguments utiles :
- `--tile-size` : Taille des tuiles (défaut : 640)
- `--val-ratio` : Proportion de validation (défaut : 0.2 = 20%)
- `--neg-ratio` : Tuiles vides par tuile positive (défaut : 1.0)
- `--force` : Efface et recrée le dataset s'il existe déjà

---

### 2. `yolo_detection.py` - Détection et Entraînement

**But :** Script principal qui fait deux choses :
1. **Entraîner** un modèle YOLO sur vos annotations
2. **Appliquer** un modèle entraîné sur de nouvelles orthophotos

#### Mode 1 : Entraînement

```bash
python yolo_detection.py train --data dataset/data.yaml --epochs 100 --imgsz 640 --base-model yolov8s.pt
```

**Ce que fait le script :**
1. **Charge un modèle pré-entraîné** : `yolov8s.pt` est un modèle qui a déjà appris sur 80 classes d'objets (COCO dataset)
2. **Transfer learning** : Il adapte ce modèle à vos 2 classes (passage_pieton, mobilier_urbain)
3. **Entraîne pendant 100 epochs** : Une "epoch" = un passage complet sur toutes les images d'entraînement
4. **Utilise l'accélération matérielle** :
   - `cuda` sur Windows/Linux avec carte NVIDIA
   - `mps` sur Mac Apple Silicon (M1/M2/M3)
   - `cpu` en dernier recours (plus lent)
5. **Augmentation des données** : `flipud=0.5` retourne verticalement 50% des images (utile en vue aérienne)
6. **Sauvegarde les meilleurs poids** : Dans `runs/detect/train/weights/best.pt`

**Arguments utiles :**
- `--epochs` : Nombre de passages (défaut : 100)
- `--imgsz` : Taille des images (défaut : 640)
- `--base-model` : Modèle de départ (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt)
- `--batch` : Nombre d'images traitées en parallèle (défaut : 16)

#### Mode 2 : Inférence (Détection)

```bash
python yolo_detection.py infer --ortho ortho.tif --weights runs/detect/train/weights/best.pt --out detections.gpkg
```

**Ce que fait le script :**
1. **Charge le modèle entraîné** : Les poids `.pt` contiennent ce que le modèle a appris
2. **Découpe l'orthophoto en tuiles** : Avec chevauchement (overlap) pour ne pas couper les objets au bord
3. **Lance la détection** sur chaque tuile
4. **Reprojette en coordonnées terrain** : Convertit les pixels en mètres/coordonnées géographiques
5. **Fusionne les doublons** : Si un passage est détecté dans plusieurs tuiles, il ne le garde qu'une fois
6. **Exporte en GeoPackage** : Un fichier SIG chargeable dans QGIS

**Arguments utiles :**
- `--tile-size` : Taille des tuiles (défaut : 640)
- `--overlap` : Chevauchement en pixels (défaut : 64)
- `--conf` : Seuil de confiance (défaut : 0.35 = 35%)
- `--epsg` : Code CRS si absent du fichier (défaut : 2154 = Lambert-93)

#### Pourquoi le chevauchement (overlap) ?
Sans chevauchement, un passage piéton coupé exactement entre deux tuiles pourrait être :
- Détecté partiellement dans la tuile de gauche
- Détecté partiellement dans la tuile de droite
- Mais jamais détecté complètement !

Avec 64 pixels de chevauchement, chaque objet a plus de chances d'être entièrement contenu dans au moins une tuile.

---

### 3. `test_yolo.py` - Script de Test

**But :** Script de test simple pour découper une orthophoto et lancer une détection rapide.

**⚠️ Note :** Ce script semble être en développement et contient quelques erreurs (chemins hardcodés, syntaxe). Il est recommandé d'utiliser `yolo_detection.py` à la place.

**Ce qu'il essaie de faire :**
1. Découper un VRT en tuiles JPEG
2. Charger un modèle YOLO entraîné
3. Lancer la prédiction sur toutes les tuiles
4. Sauvegarder les résultats avec visualisation

---

## 🚀 Workflow Complet Recommandé

### Pour créer un nouveau modèle :

1. **Préparer les annotations dans QGIS**
   - Charger l'orthophoto
   - Digitaliser les passages piétons
   - Exporter en GeoPackage avec champ `classe`

2. **Construire le dataset**
   ```bash
   python prepa_dataset.py build --ortho pcrs/ --annotations passage.gpkg --out dataset
   ```

3. **Entraîner le modèle**
   ```bash
   python yolo_detection.py train --data dataset/data.yaml --epochs 100 --imgsz 640 --base-model yolov8s.pt
   ```

4. **Tester sur une nouvelle orthophoto**
   ```bash
   python yolo_detection.py infer --ortho nouvelle_ortho.tif --weights runs/detect/train/weights/best.pt --out resultats.gpkg
   ```

5. **Ouvrir les résultats dans QGIS**
   - Charger `resultats.gpkg`
   - Les couches `passage_pieton` et `mobilier_urbain` apparaissent automatiquement

---

## 📖 Concepts Clés pour Débutants

### Système de Coordonnées (CRS)
Les orthophotos sont géoréférencées : chaque pixel correspond à une position réelle sur Terre. Le code EPSG (ex: 2154 pour Lambert-93) définit ce système de coordonnées.

### GeoPackage (.gpkg)
Format de fichier SIG qui peut contenir plusieurs couches vectorielles. C'est le format moderne recommandé pour les données géographiques.

### VRT (Virtual Raster)
Fichier texte qui fait référence à plusieurs GeoTIFF comme s'ils n'en formaient qu'un seul. Utile pour traiter des mosaïques d'images sans les fusionner physiquement.

### Transfer Learning
Au lieu d'entraîner un modèle from scratch (ce qui demanderait des milliers d'images), on part d'un modèle qui sait déjà reconnaître des formes générales et on l'adapte à nos classes spécifiques. C'est beaucoup plus rapide et nécessite moins de données.

### Augmentation de Données
Technique pour augmenter artificiellement la taille du dataset en appliquant des transformations (retournement, rotation, etc.) aux images existantes. Cela aide le modèle à généraliser mieux.

### Train vs Validation
- **Train** : Données utilisées pour enseigner au modèle
- **Validation** : Données que le modèle n'a jamais vues, utilisées pour vérifier qu'il a bien appris (et pas juste par cœur)

---

## 🔧 Dépendances Python

Le projet utilise plusieurs bibliothèques Python :

- **ultralytics** : Bibliothèque YOLOv8
- **rasterio** : Lecture/écriture de GeoTIFF
- **geopandas** : Manipulation de données géographiques
- **shapely** : Opérations géométriques
- **opencv-python (cv2)** : Traitement d'images
- **numpy** : Calculs numériques
- **pytorch** : Framework d'apprentissage profond (installé automatiquement avec ultralytics)

Pour installer :
```bash
pip install ultralytics rasterio geopandas opencv-python numpy
```

---

## 💡 Conseils pour de Bons Résultats

1. **Qualité des annotations** : Plus les annotations sont précises, meilleur sera le modèle
2. **Diversité des données** : Variez les orientations, tailles, conditions d'éclairage
3. **Équilibre des classes** : Ayez approximativement le même nombre d'exemples pour chaque classe
4. **Taille du dataset** : Au minimum 50-100 images par classe pour des résultats acceptables
5. **Sur-apprentissage** : Si le modèle est parfait sur train mais mauvais sur val, il a "appris par cœur" → augmentez les données ou réduisez la complexité du modèle

---

## 📞 Support

Pour des questions spécifiques sur :
- **QGIS** : Documentation officielle QGIS
- **YOLO/Ultralytics** : https://docs.ultralytics.com/
- **Python/SIG** : Forums GeoPython, StackOverflow

---

*Ce README est conçu pour être compréhensible par des débutants en IA et SIG. N'hésitez pas à poser des questions si certains concepts ne sont pas clairs !*
