import os
import json
import cv2
from shapely.geometry import Polygon
import time

# === Paths ===
frontImageDir = "output/front"
backImageDir = "output/back"
inputScanDir = "_INPUT"
visualOutputDir = "debug/final"

frontCoordsPath = "debug/frontCoords.json"
backCoordsPath = "debug/backCoords.json"

noCardMatches = []
noScanMatches = []

# === Ensure output dir exists ===
os.makedirs(visualOutputDir, exist_ok=True)

# === Load JSON Data ===
with open(frontCoordsPath, "r") as f:
    frontData = json.load(f)

with open(backCoordsPath, "r") as f:
    backData = json.load(f)

# === IoU-style matcher ===
def boxMatch(fx, fy, bx, by):
    half = 125
    frontRect = Polygon([
        (fx - half, fy - half),
        (fx + half, fy - half),
        (fx + half, fy + half),
        (fx - half, fy + half)
    ])
    backRect = Polygon([
        (bx - half, by - half),
        (bx + half, by - half),
        (bx + half, by + half),
        (bx - half, by + half)
    ])
    intersection = frontRect.intersection(backRect)
    return intersection.area, intersection.area >= 1000

# === Loop through scans ===
totalStart = time.time()

for frontScanKey, frontCards in frontData.items():
    scanPrefix = frontScanKey.replace("-front", "")
    backScanKey = f"{scanPrefix}-back"

    if backScanKey not in backData:
        print(f"[WARN] No matching back scan for {frontScanKey}")
        continue

    print(f"[INFO] Matching cards from {scanPrefix}...")
    backCards = backData[backScanKey]

    imagePath = os.path.join(inputScanDir, f"{scanPrefix}-front.png")
    image = cv2.imread(imagePath)

    if image is None:
        print(f"[ERROR] Could not read image: {imagePath}")
        continue

    overlay = image.copy()

    # Match cards
    for frontCardID, frontCoords in frontCards.items():
        fx, fy = frontCoords["x"], frontCoords["y"]
        bestMatch = max(
            backCards.items(),
            key=lambda item: boxMatch(fx, fy, item[1]["x"], item[1]["y"])[0],
            default=(None, None)
        )[0]

        if bestMatch is None:
            noCardMatches.append(frontCardID)
            noScanMatches.append(scanPrefix)
        else:
            area, _ = boxMatch(fx, fy, backCards[bestMatch]["x"], backCards[bestMatch]["y"])
            print(f"→ {frontCardID} ⇔ {bestMatch} (Overlap area = {area:.2f})")

    # Draw front (red) and back (blue) boxes
    for coords in frontCards.values():
        fx, fy = coords["x"], coords["y"]
        cv2.rectangle(overlay, (fx - 125, fy - 125), (fx + 125, fy + 125), (0, 0, 255), -1)

    for coords in backCards.values():
        bx, by = coords["x"], coords["y"]
        cv2.rectangle(overlay, (bx - 125, by - 125), (bx + 125, by + 125), (255, 0, 0), -1)

    # Blend and save
    blended = cv2.addWeighted(overlay, 0.4, image, 0.6, 0)
    outPath = os.path.join(visualOutputDir, f"{scanPrefix}_boxes.png")
    cv2.imwrite(outPath, blended)

# Final debug output
print(f"[DEBUG] {len(noScanMatches)} scans with 'None' matches: {noScanMatches}")
print(f"[DEBUG] {len(noCardMatches)} cards with 'None' matches: {noCardMatches}")
print(f"[COMPLETE] Matching completed in {time.time() - totalStart:.2f} seconds")
