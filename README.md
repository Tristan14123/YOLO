# Pedestrian Crossing Detection Project with YOLO

This project uses the YOLOv8 and YOLO26 artificial intelligence models from https://github.com/ultralytics/ultralytics to automatically detect pedestrian crossings in orthophotos (high-resolution aerial imagery).

The orthophotos used are 2023 PCRS aerial photographs provided by the local authority.

## 📚 Overview for Beginners

### What is YOLO?

**YOLO** (You Only Look Once) is a very popular object detection algorithm. Unlike older methods that analyze an image multiple times, YOLO looks at the image only once to detect all objects simultaneously. This is what makes it so fast!

### Why split orthophotos?

PCRS orthophotos (Very High Resolution, 5 cm/pixel) are huge files (several GB) with thousands of pixels on each side. YOLO cannot process such a large image all at once. Therefore, it is necessary to:

1. **Split** the orthophoto into small pieces (tiles) of 640x640 pixels
2. **Analyze** each tile with YOLO
3. **Reconstruct** the results using real-world geographic coordinates

---

## 🛠️ Python Scripts

### 1. `prepa_dataset.py` — Dataset Preparation

**Purpose:** Prepare training data for YOLO from orthophotos and annotations created in QGIS.

#### What is a training dataset?

For an AI model to learn how to recognize pedestrian crossings, it needs to be shown examples. A dataset contains:

* **Images**: pieces of orthophotos (tiles)
* **Labels**: text files indicating where pedestrian crossings are located in each image
* **Train/val split**:

  * `train` (80%): used to teach the AI
  * `val` (20%): used to test whether the AI has learned correctly

#### Complete workflow:

##### Step 1: Digitizing in QGIS

1. Load the orthophoto in QGIS
2. Create a vector layer (GeoPackage) with a `classe` field
3. Manually draw:

   * **passage_pieton** → as a POLYGON around the pedestrian crossing
   * **mobilier_urbain** → as a POINT on the object

##### Step 2: Building the dataset

```bash
python prepa_dataset.py build --ortho pcrs_tout.vrt --annotations passage.gpkg --out dataset
```

See `passage.gpkg` for the model training areas.

**What the script does:**

1. **Lists the tiles**: It accepts a GeoTIFF, a VRT (virtual mosaic), or a tile directory
2. **Plans intelligently**: It only reads tiles containing annotations (saving processing time!)
3. **Splits into tiles**: It divides each tile into 640x640 pixel tiles
4. **Converts annotations**: It converts QGIS geographic coordinates into YOLO pixel coordinates
5. **Selects useful tiles**:

   * Keeps **all** tiles containing pedestrian crossings (positive samples)
   * Adds a **sample** of empty tiles (negative samples) so the model learns not to detect objects everywhere
6. **Intelligent spatial split**: It prevents neighboring tiles belonging to the same crossing from being split between training and validation
7. **Creates the following files**:

   * `dataset/images/train/*.jpg` — Training images
   * `dataset/images/val/*.jpg` — Validation images
   * `dataset/labels/train/*.txt` — Training labels
   * `dataset/labels/val/*.txt` — Validation labels
   * `dataset/data.yaml` — YOLO configuration file
   * `dataset/manifest.csv` — Inventory of all tiles

**⚠️ Important:** In every tile used, annotate ALL visible pedestrian crossings. The "empty" tiles are used as negative examples — an unannotated pedestrian crossing would be learned as "this is not a pedestrian crossing"!

#### Useful arguments:

* `--tile-size`: Tile size (default: 640)
* `--val-ratio`: Validation proportion (default: 0.2 = 20%)
* `--neg-ratio`: Empty tiles per positive tile (default: 1.0)
* `--force`: Deletes and recreates the dataset if it already exists

---

### 2. `yolo_detection.py` — Detection and Training

**Purpose:** Main script that does two things:

1. Train a YOLO model using your annotations
2. Apply a trained model to new orthophotos

#### Mode 1: Training

```bash
python yolo_detection.py train --data dataset/data.yaml --epochs 100 --imgsz 640 --base-model yolov8s.pt
```

**What the script does:**

1. **Loads a pre-trained model**: `yolov8s.pt` is a model that has already learned 80 object classes from the COCO dataset
2. **Transfer learning**: It adapts this model to your 2 classes (`passage_pieton`, `mobilier_urbain`)
3. **Trains for 100 epochs**: An "epoch" is one complete pass through all training images
4. **Uses hardware acceleration**:

   * `cuda` on Windows/Linux with an NVIDIA graphics card
   * `mps` on Mac Apple Silicon (M1/M2/M3)
   * `cpu` as a last resort (slower)
5. **Data augmentation**: `flipud=0.5` vertically flips 50% of the images (useful for aerial imagery)
6. **Saves the best weights**: In `runs/detect/train/weights/best.pt`

#### Useful arguments:

* `--epochs`: Number of training passes (default: 100)
* `--imgsz`: Image size (default: 640)
* `--base-model`: Starting model (`yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`)
* `--batch`: Number of images processed simultaneously (default: 16)

#### Mode 2: Inference (Detection)

```bash
python yolo_detection.py infer --ortho ortho.tif --weights runs/detect/train/weights/best.pt --out detections.gpkg
```

**What the script does:**

1. **Loads the trained model**: The `.pt` weights contain what the model has learned
2. **Splits the orthophoto into tiles**: With overlap to prevent objects from being cut off at tile boundaries
3. **Runs detection** on each tile
4. **Reprojects the results into real-world coordinates**: Converts pixels into meters/geographic coordinates
5. **Merges duplicates**: If a crossing is detected in several tiles, it is kept only once
6. **Exports to GeoPackage**: A GIS file that can be loaded into QGIS

#### Useful arguments:

* `--tile-size`: Tile size (default: 640)
* `--overlap`: Overlap in pixels (default: 64)
* `--conf`: Confidence threshold (default: 0.35 = 35%)
* `--epsg`: CRS code if missing from the file (default: 2154 = Lambert-93)

### Why use overlap?

Without overlap, a pedestrian crossing located exactly between two tiles could be:

* Partially detected in the left tile
* Partially detected in the right tile
* But never detected completely!

With a 64-pixel overlap, each object is more likely to be fully contained within at least one tile.

---

### 3. `yolo_detection.py` — Test Script

**Purpose:** A simple test script for splitting an orthophoto and running a quick detection.

**What it does:**

1. Splits a VRT into JPEG tiles
2. Loads a trained YOLO model
3. Runs prediction on all tiles
4. Saves the results with visualization

See:

---

## 🚀 Recommended Complete Workflow

### To create a new model:

1. **Prepare the annotations in QGIS**

   * Load the orthophoto
   * Digitize the pedestrian crossings
   * Export as a GeoPackage with a `classe` field

2. **Build the dataset**

   ```bash
   python prepa_dataset.py build --ortho pcrs/ --annotations passage.gpkg --out dataset
   ```

3. **Train the model**

   ```bash
   python yolo_detection.py train --data dataset/data.yaml --epochs 100 --imgsz 640 --base-model yolov8s.pt
   ```

4. **Test on a new orthophoto**

   ```bash
   python yolo_detection.py infer --ortho nouvelle_ortho.tif --weights runs/detect/train/weights/best.pt --out test.gpkg
   ```

5. **Open the results in QGIS**

   * Load `test.gpkg`
   * The `passage_pieton` and `mobilier_urbain` layers will appear automatically

---

## 📖 Key Concepts for Beginners

### Coordinate Reference System (CRS)

Orthophotos are georeferenced: each pixel corresponds to a real-world position on Earth. The EPSG code (e.g. 2154 for Lambert-93) defines this coordinate reference system.

### GeoPackage (.gpkg)

A GIS file format that can contain multiple vector layers. It is the modern recommended format for geographic data.

### VRT (Virtual Raster)

A text file that references several GeoTIFF files as if they formed a single dataset. It is useful for processing image mosaics without physically merging them.

### Transfer Learning

Instead of training a model from scratch (which would require thousands of images), we start with a model that already knows how to recognize general shapes and adapt it to our specific classes. This is much faster and requires less data.

### Data Augmentation

A technique used to artificially increase the size of a dataset by applying transformations (flipping, rotation, etc.) to existing images. This helps the model generalize better.

### Train vs Validation

* **Train**: Data used to teach the model
* **Validation**: Data the model has never seen, used to check whether it has learned correctly rather than simply memorizing the training data

---

## 🔧 Python Dependencies

The project uses several Python libraries:

* **ultralytics**: YOLOv8 library
* **rasterio**: GeoTIFF reading/writing
* **geopandas**: Geographic data manipulation
* **shapely**: Geometric operations
* **opencv-python (cv2)**: Image processing
* **numpy**: Numerical calculations
* **pytorch**: Deep learning framework (automatically installed with ultralytics)

To install:

```bash
pip install ultralytics rasterio geopandas opencv-python numpy
```

---

## 💡 Tips for Good Results

1. **Annotation quality**: The more accurate the annotations, the better the model
2. **Data diversity**: Vary orientations, sizes, and lighting conditions
3. **Class balance**: Have approximately the same number of examples for each class
4. **Dataset size**: At least 50–100 images per class for acceptable results
5. **Overfitting**: If the model is perfect on the training set but performs poorly on validation, it has "memorized" the data → increase the amount of data or reduce model complexity

---

## 📞 Support

For specific questions about:

* **QGIS**: Official QGIS documentation
* **YOLO/Ultralytics**: https://docs.ultralytics.com/
* **Python/GIS**: GeoPython forums, StackOverflow

---

*This README is designed to be understandable for beginners in AI and GIS. Feel free to ask questions if some concepts are unclear!*

This project was developed as part of the validation of my studies. It addresses the following issue:

In order to facilitate future updates of accessibility data, the local authority wants to investigate the feasibility of change detection based on 5 cm orthophotographs regularly produced across its territory. In particular, detecting pedestrian crossings (newly created or removed) would be relevant, as would detecting street furniture.

You are expected to conduct a study of feasible AI tools/methods, with an estimation of the confidence level and completeness of the data produced by AI.

# PROJECT HISTORY

## PROJECT BACKGROUND AND DEVELOPMENT HISTORY

This project was developed as part of the validation of my academic studies and addresses the following operational issue:

> In order to facilitate future updates to accessibility-related data, the local authority wishes to investigate the feasibility of detecting changes based on 5 cm orthophotographs regularly produced across its territory. In particular, the automatic detection of pedestrian crossings, including newly created or removed crossings, could support the updating of accessibility data. The detection of street furniture is also considered relevant.

The objective of the project was therefore to assess the feasibility of using artificial intelligence methods for the automated detection of these features. The study was also intended to provide an assessment of the **confidence and completeness of the data generated by the AI-based approach**.

### Initial approach: QGIS Deepness

An initial experiment was conducted using the **Deepness extension for QGIS**, with the objective of evaluating an existing AI-based image detection workflow within a GIS environment.

However, the experiment did not produce satisfactory results. In addition to the limitations encountered during the detection process, the installation and configuration of the extension were complicated by the coexistence of several Python environments and versions used by QGIS and other projects.

As a result, an alternative approach was considered in order to have greater control over the processing environment and the different stages of the workflow. The decision was therefore made to develop dedicated Python scripts for dataset preparation, model training and inference.

### Selection of YOLO

YOLO was selected as the main object-detection framework based on the availability of documentation and reported results for real-time object detection. This approach was preferred to a TensorFlow-based workflow for the purposes of this project.

An initial test was conducted using a model previously trained for pedestrian-crossing detection in an embedded-detection context. However, this model did not produce positive detections on the aerial imagery used in this project.

This result highlighted the difference between the imagery used to train an existing model and the **5 cm aerial orthophotography** used in this study. Consequently, a dedicated model was trained using aerial imagery representative of the project area.

### Training of the YOLOv8s model

The **YOLOv8s** model was subsequently trained specifically to detect pedestrian crossings in aerial imagery.

A sample of **269 areas** was defined across three PCRS tiles covering different types of environments in Granville:

* the city center;
* the peripheral or less densely built-up urban area;
* the surrounding rural area.

This sampling strategy was intended to expose the model to different spatial and environmental configurations.

Following the training phase, the model was evaluated on an orthophoto that had **not been used during training**. This made it possible to assess the detection process on previously unseen imagery.

The visual results obtained during testing were considered convincing. However, the processing speed was limited by the available hardware.

### Comparison with YOLO26n

To investigate an alternative model architecture and assess its performance under the same conditions, the **YOLO26n** model was subsequently trained and tested using the same general methodology.

The resulting detections were also visually convincing. However, in the tested configuration, the YOLO26n model processed the imagery at approximately **one-third of the speed** observed with the YOLOv8s workflow.

This comparison illustrates the trade-off between detection performance and computational efficiency that must be considered when designing an operational workflow for large-scale orthophoto processing.

### Development of the processing workflow

The final workflow was therefore designed around a dedicated Python processing chain rather than a QGIS plugin. It covers the main stages required for the experiment:

1. preparation of the training dataset from georeferenced annotations;
2. extraction of image tiles from large orthophotos;
3. conversion of geographic annotations into YOLO-compatible labels;
4. training of the object-detection model;
5. inference on previously unseen orthophotos;
6. conversion of detections back into geographic coordinates;
7. export of the detected objects as GIS data in GeoPackage format.

This approach provides a link between the artificial intelligence detection process and the existing GIS workflow, allowing the generated results to be subsequently inspected and processed in QGIS.

### Use of artificial intelligence during development

Part of the Python scripts was developed with assistance from **Claude Sonnet 4.6**. The use of AI assistance mainly supported the development and structuring of the scripts used in the processing workflow.

The resulting workflow was subsequently tested using the project's own orthophotographic data, annotations and trained models.
