# Postcard Digitization and Matching Tool

This project is a collection of Python scripts that automate the digitization of physical postcards. It is optimized for batch processing with speed, consistency, and minimal human interaction.

## 🔧 Features

- Automatically detects and crops postcards from raw `.png` scan images.
- Matches front and back postcard images using geometric overlap logic.
- Saves debug images with colored overlays:
  - ✅ **Green**: Accepted/Matched cards
  - 🔴 **Red**: Front-only detections
  - 🔵 **Blue**: Back-only detections
  - ⚪ **White**: All detected closed contours
- Persistent caching of coordinate and contour data for efficient re-runs.
- Configurable output and debug folder structure.
- Parallelized processing for speed.

## 🚀 How to Use

### 1. Prepare Your Scans

Place your raw scanned `.png` files into the `_INPUT/` directory.
Each scan should be named consistently:

```
sc#-front.png
sc#-back.png
```

For example:

```
sc7-front.png
sc7-back.png
```

### 2. Run the Scripts

You can run each script manually:

```bash
python3 front-scanner-v4.py   # Processes front images
python3 back-scanner-v4.py    # Processes back images
python3 combine-images-v3.py  # Matches fronts with backs and saves debug overlays
```

Or, run the master script:

```bash
python3 scan-master.py        # Threaded script to run everything at once
```

## 📁 Review Output

- Cropped postcard images:
  - `output/front/`
  - `output/back/`
- Visual match overlays and contour debugging:
  - `debug/final/`

## ⚙️ Dependencies

- Python 3.8+
- OpenCV
- NumPy
- Shapely
- tqdm

Install them with:

```bash
pip install opencv-python shapely numpy tqdm
```

## 📌 Notes

- `_SIMPLE/` is used for intermediate images post-cleanup.
- Scripts are optimized for long, rectangular postcards on a black scan background.
- The project assumes each scan has a *front* and *back* counterpart.
