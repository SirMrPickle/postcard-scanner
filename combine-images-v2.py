import os
import json
import cv2
import numpy as np
from shapely.geometry import Polygon

# === Paths ===
frontImageDir = "output/front"
backImageDir = "output/back"
inputScanDir = "_INPUT"
visualOutputDir = "debug/final"

frontCoordsPath = "debug/front_coords.json"
backCoordsPath = "debug/back_coords.json"

noMatches = []

# === Ensure output dir exists ===
os.makedirs(visualOutputDir, exist_ok=True)

# === Load JSON Data ===
with open(frontCoordsPath, "r") as f:
    frontData = json.load(f)

with open(backCoordsPath, "r") as f:
    backData = json.load(f)

# === IoU-style matcher ===
def boxMatch(fx, fy, bx, by):
    targetArea = 250  # px
    half = targetArea / 2

    frontRectangle = Polygon([
        (fx - half, fy - half),
        (fx + half, fy - half),
        (fx + half, fy + half),
        (fx - half, fy + half),
        (fx - half, fy - half),
    ])

    backRectangle = Polygon([
        (bx - half, by - half),
        (bx + half, by - half),
        (bx + half, by + half),
        (bx - half, by + half),
        (bx - half, by - half),
    ])

    intersection = frontRectangle.intersection(backRectangle)
    isValid = intersection.area >= 1000

    return intersection.area, isValid

# === Loop through scans ===
for frontScanKey in frontData:
    scanPrefix = frontScanKey.replace("-front", "")
    backScanKey = f"{scanPrefix}-back"

    if backScanKey not in backData:
        print(f"[WARN] No matching back scan for {frontScanKey}")
        continue

    print(f"[INFO] Matching cards from {scanPrefix}...")

    frontCards = frontData[frontScanKey]
    backCards = backData[backScanKey]

    # Load the original front image
    inputImagePath = os.path.join(inputScanDir, f"{scanPrefix}-front.png")
    image = cv2.imread(inputImagePath)

    if image is None:
        print(f"[ERROR] Could not read image: {inputImagePath}")
        continue

    overlay = image.copy()

    for frontCardID, frontCoords in frontCards.items():
        fx, fy = frontCoords["x"], frontCoords["y"]
        bestMatch = None
        bestScore = 0

        for backCardID, backCoords in backCards.items():
            bx, by = backCoords["x"], backCoords["y"]
            area, isValid = boxMatch(fx, fy, bx, by)

            if area > bestScore:
                bestScore = area
                bestMatch = backCardID

        if bestMatch is None:
            noMatches.append(frontCardID)
        print(f"→ {frontCardID} ⇔ {bestMatch} (Overlap area = {bestScore:.2f})")

        # Draw front box in red
        frontTopLeft = (fx - 125, fy - 125)
        frontBottomRight = (fx + 125, fy + 125)
        cv2.rectangle(overlay, frontTopLeft, frontBottomRight, (0, 0, 255), thickness=-1)

        # Draw matched back box in blue if it exists
        if bestMatch and bestMatch in backCards:
            bx, by = backCards[bestMatch]["x"], backCards[bestMatch]["y"]
            backTopLeft = (bx - 125, by - 125)
            backBottomRight = (bx + 125, by + 125)
            cv2.rectangle(overlay, backTopLeft, backBottomRight, (255, 0, 0), thickness=-1)

    # Blend original image and overlay
    alpha = 0.4
    output = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # Save output
    outputPath = os.path.join(visualOutputDir, f"{scanPrefix}_boxes.png")
    cv2.imwrite(outputPath, output)

print(f"[DEBUG] Cards with `None` matches: {noMatches}")
