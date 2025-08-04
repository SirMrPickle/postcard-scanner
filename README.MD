# Postcard Digitization and Matching Tool

A high-throughput Python-based pipeline for digitizing, cropping, and matching physical postcard scans. Built for speed, consistency, and minimal manual intervention — optimized for batch processing and AWS-scale extension.

## 🔧 Features

- 🔍 **Auto-detects** and crops postcard regions from raw `.png` scans (front and back).
- 🧠 **Smart matching** of front and back images based on contour geometry (overlap area, center point proximity).
- 🖼️ Saves debug overlays for visual QA:
  - ✅ **Green**: Successful front-back matches
  - 🔴 **Red**: Orphaned front detections
  - 🔵 **Blue**: Orphaned back detections
  - ⚪ **White**: All closed contours detected
- 🧠 **Warping & alignment** based on detected quadrilateral box
- 💾 **Persistent caching** of contour and box data in JSON
- 🧵 **Threaded matching** for faster execution
- 💻 Built to support AWS / SageMaker pipelines (e.g., Textract + RDS)

## 🚀 How to Use

### 1. Drop Your Scans

Place your raw `.png` scans into the `_INPUT/` directory with consistent naming:

sc#-front.png
sc#-back.png

For example:

sc7-front.png
sc7-back.png

### 2. Run the Pipeline

**Manual Steps (per-stage debugging):**

python3 front-scanner-v4.py   # Detects & crops fronts
python3 back-scanner-v4.py    # Detects & crops backs
python3 combine-v5.py         # Matches front↔back and saves combined results

**All-in-One Execution:**

python3 scan-master.py        # End-to-end threaded pipeline

## 📁 Output Structure

### ✅ Results

- output/front/ — Cropped front images
- output/back/ — Cropped back images
- output/matched/ — Combined front-back postcards

### 🧪 Debugging & QA

- debug/frontCoords.json — Cropping data (fronts)
- debug/backCoords.json — Cropping data (backs)
- debug/final/ — Matching overlays & composite debug images
- debug/contourData.txt — All contour metadata
- debug/cardContours.png — All card-sized contours
- debug/topContours.png — Top-ranked card shapes
- debug/closedBoxes.png — Final accepted boxes

## ⚙️ Requirements

- Python 3.8+
- opencv-python
- numpy
- shapely
- tqdm

Install them with:

``pip install opencv-python numpy shapely tqdm``

## 📌 Notes & Tips

- This tool assumes your cards are:
  - Scanned on a solid **black** background
  - Rectangular and longer on one edge

## 🛠️ Disclamer

This markdown file was created with the help of AI.
