import os
import json
import cv2
from shapely.geometry import Polygon
import time
import numpy as np
import pytesseract

"""
Time Estimations:
30.86/61 
    = 0.5059016393 Seconds/Combine
"""
# === Paths ===
frontImageDir = "output/front"
backImageDir = "output/back"
inputScanDir = "_INPUT"
visualOutputDir = "debug/final"
outputDir = "output/final"


frontCoordsPath = "debug/frontCoords.json"
backCoordsPath = "debug/backCoords.json"

noCardMatches = []
noScanMatches = []

# === For image saving and stacking ===
DPI = 250
WIDTH = int(8.5 * DPI)  # 8.5x11in sheet as pixels
HEIGHT = int(11 * DPI)

# === Ensure output dir exists ===
os.makedirs(visualOutputDir, exist_ok=True)
os.makedirs(outputDir, exist_ok=True)

# === Load JSON Data ===
with open(frontCoordsPath, "r") as f:
    frontData = json.load(f)

with open(backCoordsPath, "r") as f:
    backData = json.load(f)


# === IoU-style matcher ===
def boxMatch(fx, fy, bx, by):
    half = 125
    frontRect = Polygon(
        [
            (fx - half, fy - half),
            (fx + half, fy - half),
            (fx + half, fy + half),
            (fx - half, fy + half),
        ]
    )
    backRect = Polygon(
        [
            (bx - half, by - half),
            (bx + half, by - half),
            (bx + half, by + half),
            (bx - half, by + half),
        ]
    )
    intersection = frontRect.intersection(backRect)
    return intersection.area, intersection.area >= 1000


def horizontalOrient(image):
    if image.shape[0] > image.shape[1]:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image


def pad(image, width=WIDTH, height=HEIGHT):
    h, w = image.shape[:2]

    # Resize image if too big
    if h > height or w > width:
        scale = min(width / w, height / h)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
        h, w = image.shape[:2]

    # Create light white noise background
    background = np.random.randint(43, 47, (height, width, 3), dtype=np.uint8)

    # Compute padding offsets
    padTop = (height - h) // 2
    padLeft = (width - w) // 2

    # Paste image onto background
    background[padTop : padTop + h, padLeft : padLeft + w] = image
    return background

def getImageOrientation(image):
    try:
        osd = pytesseract.image_to_osd(image)
        for line in osd.split('\n'):
            if 'Rotate:' in line:
                return int(line.split(':')[1].strip())
    except pytesseract.TesseractError as e:
        print(f"[WARN] OSD failed: {e}. Defaulting rotation to 0.")
        return 0
    return 0

def rotateImage(image, angle):
    if angle == 0:
        return image
    elif angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    else:
        return image  # fallback


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
            default=(None, None),
        )[0]

        if bestMatch is None:
            noCardMatches.append(frontCardID)
            noScanMatches.append(scanPrefix)
        else:
            bx, by = backCards[bestMatch]["x"], backCards[bestMatch]["y"]
            area, _ = boxMatch(fx, fy, bx, by)
            print(f"→ {frontCardID} ⇔ {bestMatch} (Overlap area = {area:.2f})")

            # track bad matches as well as plain old `none`s
            if area < 10000:
                noCardMatches.append(frontCardID)
                noScanMatches.append(scanPrefix)

        if bestMatch is not None and area >= 10000:
            # load back the images again
            frontCardPath = os.path.join(frontImageDir, f"{frontCardID}_front.png")
            backCardPath = os.path.join(backImageDir, f"{bestMatch}_back.png")

            frontImage = cv2.imread(frontCardPath)
            backImage = cv2.imread(backCardPath)

            if frontImage is None or backImage is None:
                print(
                    f"[WARN] Missing front or back card image for {frontCardID} / {bestMatch}"
                )  # i <3 debugging
                continue

            frontImage = horizontalOrient(frontImage)
            backImage = horizontalOrient(backImage)
            
            
            '''
            This doesnt quite work right. To fix later
            frontImage = rotateImage(frontImage, getImageOrientation(frontImage))
            backImage = rotateImage(backImage, getImageOrientation(backImage))
            '''
            
            # Stack vertically, centered horizontally
            stackedHeight = frontImage.shape[0] + backImage.shape[0]
            stackedWidth = max(frontImage.shape[1], backImage.shape[1])

            front_x = (stackedWidth - frontImage.shape[1]) // 2
            back_x = (stackedWidth - backImage.shape[1]) // 2

            stackedImage = np.full((stackedHeight, stackedWidth, 3), 255, dtype=np.uint8)  # white bg

            stackedImage[0:frontImage.shape[0], front_x:front_x+frontImage.shape[1]] = frontImage
            stackedImage[frontImage.shape[0]:, back_x:back_x+backImage.shape[1]] = backImage

            # Pad stacked image to 8.5x11
            finalImage = pad(stackedImage)

            # Create noise background
            background = np.random.randint(43, 47, (HEIGHT, WIDTH, 3), dtype=np.uint8)

            # Replace pure white pixels with noise background pixels
            mask = (finalImage == 255).all(axis=2)
            finalImage[mask] = background[mask]

            # Save final combined image using front card name
            outFilePath = os.path.join(outputDir, f"{frontCardID}.png")
            cv2.imwrite(outFilePath, finalImage)


    # Draw front (red) and back (blue) boxes
    for coords in frontCards.values():
        fx, fy = coords["x"], coords["y"]
        cv2.rectangle(
            overlay, (fx - 125, fy - 125), (fx + 125, fy + 125), (0, 0, 255), -1
        )

    for coords in backCards.values():
        bx, by = coords["x"], coords["y"]
        cv2.rectangle(
            overlay, (bx - 125, by - 125), (bx + 125, by + 125), (255, 0, 0), -1
        )

    # Blend and save
    blended = cv2.addWeighted(overlay, 0.4, image, 0.6, 0)
    outPath = os.path.join(visualOutputDir, f"{scanPrefix}_boxes.png")
    cv2.imwrite(outPath, blended)

# Deduplicate <- goated word
noScanMatches = sorted(set(noScanMatches))
noCardMatches = sorted(set(noCardMatches))

print(f"\n[DEBUG] {len(noScanMatches)} scans with weak matches: {noScanMatches}")
print(f"[DEBUG] {len(noCardMatches)} cards with weak matches: {noCardMatches}")
print(f"[COMPLETE] Matching completed in {time.time() - totalStart:.2f} seconds")
