import cv2
import numpy as np
import os
import glob
import re
import json
import time

START = time.time()
# ======= FRONT SCANNER ======= #

"""
Time Estimations:

f- 78.84/61 = 1.2924590164
b- 77.77/61 = 1.2749180328
avg: 1.2836885246 Seconds/Scan
    =(78.84/61+77.77/61)/2    
"""

inputDir = "_INPUT"
outputDir = "output/front"
debugBaseDir = "debug/front"
seenPath = "counters/scannedFront.txt"
indexPath = "counters/indexFront.txt"
contourDebugPath = "debug/contoursFront.txt"
contourCoordsPath = "debug/frontCoords.json"

logDebug = False
padSize = 20
resizeFactor = 0.75
consolePrintAll = True
timeDebug = False

z, t = 30, 55
lowerGray = np.array([z, z, z])
upperGray = np.array([t, t, t])

# === DIRECTORY SETUP ===
os.makedirs(outputDir, exist_ok=True)
os.makedirs(debugBaseDir, exist_ok=True)
os.makedirs(os.path.dirname(seenPath), exist_ok=True)

if os.path.exists(seenPath):
    with open(seenPath, "r") as f:
        processedFiles = set(f.read().splitlines())
else:
    processedFiles = set()

if os.path.exists(indexPath):
    with open(indexPath, "r") as f:
        index = int(f.read().strip())
else:
    index = 1

if os.path.exists(contourCoordsPath):
    with open(contourCoordsPath, "r") as f:
        contourCoords = json.load(f)
else:
    contourCoords = {}

if os.path.exists(contourDebugPath):
    with open(contourDebugPath, "r") as f:
        finalContoursDebug = f.read().splitlines()
else:
    finalContoursDebug = []


# === FUNCTIONS ===
def extractNumber(filename):
    match = re.search(r"sc(\d+)[_-]front", filename.lower())
    return int(match.group(1)) if match else float("inf")


inputFiles = sorted(glob.glob(os.path.join(inputDir, "*.png")), key=extractNumber)
totalStart = time.time()

for inputPath in inputFiles:
    baseName = os.path.splitext(os.path.basename(inputPath))[0]

    # Skip already processed
    if baseName in processedFiles:
        print(f"[SKIP] {baseName} already processed")
        continue

    # Only process fronts
    if "front" not in inputPath.lower():
        continue

    print(f"\n[PROCESSING] {inputPath}")
    startTime = time.time()
    image = cv2.imread(inputPath)
    if image is None:
        print(f"[ERROR] Cannot open {inputPath}, skipping.")
        continue

    debugDir = os.path.join(debugBaseDir, baseName)
    os.makedirs(debugDir, exist_ok=True)

    # Pad image with noise
    if timeDebug:
        padT = time.time()
    h, w = image.shape[:2]
    padded = np.random.randint(
        z, t, (h + 2 * padSize, w + 2 * padSize, 3), dtype=np.uint8
    )
    padded[padSize : padSize + h, padSize : padSize + w] = image
    image = padded.copy()
    if timeDebug:
        print(f"[TIME] Padding took: {time.time() - padT:.4}s")

    # Mask gray background
    if timeDebug:
        maskT = time.time()
    grayMask = cv2.inRange(image, lowerGray, upperGray)
    nonBgMask = cv2.bitwise_not(grayMask)
    maskedImage = cv2.bitwise_and(image, image, mask=nonBgMask)
    if timeDebug:
        print(f"[TIME] Masking took: {time.time() - maskT:.4}s")

    # Resize and preprocess
    if timeDebug:
        preT = time.time()
    resized = cv2.resize(maskedImage, (0, 0), fx=resizeFactor, fy=resizeFactor)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    if timeDebug:
        print(f"[TIME] Preprocessing took: {time.time() - preT:.4}s")

    # Find contours
    if timeDebug:
        contourT = time.time()
    contours, hierarchy = cv2.findContours(
        closed.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    print(f"[INFO] Found {len(contours)} contours")
    if timeDebug:
        print(f"[TIME] Contour detection took: {time.time() - contourT:.4}s")

    # Filter contours based on area, aspect, and hierarchy
    filteredContours = []
    areaDebugInfo = []

    for cnt, h in zip(contours, hierarchy[0]):
        parent = h[3]
        x, y, wBox, hBox = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        aspect = wBox / hBox if hBox != 0 else 0
        areaDebugInfo.append((area, aspect, parent, (x, y, wBox, hBox)))

        if parent == -1 and area > 40000 and 0.59 < aspect < 3.0:
            filteredContours.append(cnt)

    # Sort and limit to top 6 postcard contours
    postcardContours = sorted(filteredContours, key=cv2.contourArea, reverse=True)[:6]
    print(f"[INFO] Filtered to {len(postcardContours)} candidate contours")

    # ALWAYS save cardContours.png — the top (not always 6) strongest contours
    if timeDebug:
        saveT = time.time()
    cardContoursDebug = image.copy()
    for i, cnt in enumerate(postcardContours):
        scaledCnt = (cnt / resizeFactor).astype(np.int32)
        x, y, wBox, hBox = cv2.boundingRect(scaledCnt)
        cv2.rectangle(cardContoursDebug, (x, y), (x + wBox, y + hBox), (0, 255, 0), 2)
        cv2.putText(
            cardContoursDebug,
            str(i),
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
    cv2.imwrite(os.path.join(debugDir, "cardContours.png"), cardContoursDebug)
    if timeDebug:
        print(f"[TIME] Saving debug contour image took: {time.time() - saveT:.4}s")

    # Save debug images only if fewer than 6 postcard contours found. Assumes that 6 is the propper number.
    if len(postcardContours) < 6:
        cv2.imwrite(os.path.join(debugDir, "closedBoxes.png"), closed)

        # Draw and save `topContours.png` (top 10 largest contours)
        topContours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        topContoursDebug = resized.copy()
        for i, cnt in enumerate(topContours):
            x, y, wBox, hBox = cv2.boundingRect(cnt)
            cv2.rectangle(
                topContoursDebug, (x, y), (x + wBox, y + hBox), (255, 0, 255), 2
            )
            cv2.putText(
                topContoursDebug,
                f"#{i} A={int(cv2.contourArea(cnt))}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 255),
                2,
            )
        cv2.imwrite(os.path.join(debugDir, "topContours.png"), topContoursDebug)

        # save contour info/data
        with open(os.path.join(debugDir, "contourData.txt"), "w") as f:
            for i, (area, aspect, parent, box) in enumerate(areaDebugInfo):
                f.write(
                    f"Contour {i}: Area={area:.2f}, Aspect={aspect:.2f}, Parent={parent}, Box={box}\n"
                )

    # update contourCoords with centroid info
    if timeDebug:
        coordT = time.time()
    contourCoords[baseName] = {}
    cardNumForFile = 0
    for cnt in postcardContours:
        scaledCnt = (cnt / resizeFactor).astype(np.int32)
        M = cv2.moments(scaledCnt)
        cX = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
        cY = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0

        cardNumForFile += 1
        cardName = f"card{index + cardNumForFile - 1:04d}"
        contourCoords[baseName][cardName] = {"x": cX, "y": cY}
    if timeDebug:
        print(f"[TIME] Contour detection took: {time.time() - coordT:.4}s")

    # save each postcard (warped) image
    if timeDebug:
        saveRT = time.time()
    savedCount = 0
    for cnt in postcardContours:
        scaledCnt = (cnt / resizeFactor).astype(np.int32)
        rect = cv2.minAreaRect(scaledCnt)
        box = cv2.boxPoints(rect).astype(np.intp)
        width, height = int(rect[1][0]), int(rect[1][1])

        if width == 0 or height == 0:
            print("[WARN] Skipping contour with zero width/height")
            continue

        srcPts = box.astype("float32")
        dstPts = np.array(
            [[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]],
            dtype="float32",
        )
        M = cv2.getPerspectiveTransform(srcPts, dstPts)
        warped = cv2.warpPerspective(image, M, (width, height))

        # Rotate cards to landscape
        if warped.shape[0] > warped.shape[1]:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

        outName = f"card{index:04d}_front.png"
        cv2.imwrite(os.path.join(outputDir, outName), warped)
        if consolePrintAll:
            print(f"[SAVED] {outName}")
        index += 1
        savedCount += 1
        # print(f"[DEBUG] Box size: {width}x{height}")
        if timeDebug:
            print(f"[TIME] Contour detection took: {time.time() - saveRT:.4}s")

    # Mark files and log
    processedFiles.add(baseName)
    finalContoursDebug.append(f"{baseName}: {savedCount}")
    timing = time.time() - startTime
    print(f"[DONE] {baseName} in {timing:.2}s")

# === SAVE FILES ===
with open(seenPath, "w") as f:
    f.write("\n".join(sorted(processedFiles)))

with open(indexPath, "w") as f:
    f.write(str(index))

with open(contourDebugPath, "w") as f:
    f.write("\n".join(finalContoursDebug))

with open(contourCoordsPath, "w") as f:
    json.dump(contourCoords, f, indent=2)

print(f"\n[COMPLETE] All front scans processed in {time.time() - totalStart:.2f}s")
print(f"Raw time: {time.time()-START:.2f}s")
