import cv2
import numpy as np
import os
import glob
import re
import json

# ======= FRONT SCANNER (FORCED 6-CONTOUR MODE) ======= #

inputDir = "_INPUT"
outputDir = "output/front"
debugBaseDir = "debug/front"
seenPath = "counters/scannedFront.txt"
indexPath = "counters/indexFront.txt"
contourDebugPath = "debug/contoursFront.txt"
contourCoordsPath = "debug/frontCoords.json"

logDebug = False
padSize = 20  # noise padding around edges

z, t = 30, 55  # background gray range
lowerGray = np.array([z, z, z])
upperGray = np.array([t, t, t])

os.makedirs(outputDir, exist_ok=True)
os.makedirs(debugBaseDir, exist_ok=True)
os.makedirs(os.path.dirname(seenPath), exist_ok=True)

# Load seen files
if os.path.exists(seenPath):
    with open(seenPath, "r") as f:
        processedFiles = set(f.read().splitlines())
else:
    processedFiles = set()

# Load index
if os.path.exists(indexPath):
    with open(indexPath, "r") as f:
        index = int(f.read().strip())
else:
    index = 1

# Load or initialize contour coordinates
if os.path.exists(contourCoordsPath):
    with open(contourCoordsPath, "r") as f:
        contourCoords = json.load(f)
else:
    contourCoords = {}


# Helper to extract scan number for proper order
def extractNumber(filename):
    match = re.search(r"sc(\d+)[_-]front", filename.lower())
    return int(match.group(1)) if match else float("inf")


# Sort input files correctly
inputFiles = sorted(glob.glob(os.path.join(inputDir, "*.png")), key=extractNumber)
finalContoursDebug = []

for inputPath in inputFiles:
    baseName = os.path.splitext(os.path.basename(inputPath))[0]

    if baseName in processedFiles:
        print(f"[SKIP] {baseName} already processed")
        continue

    if "front" not in inputPath.lower():
        continue

    print(f"\n[PROCESSING] {inputPath}")
    image = cv2.imread(inputPath)
    if image is None:
        print(f"[ERROR] Cannot open {inputPath}, skipping.")
        continue

    debugDir = os.path.join(debugBaseDir, baseName)
    os.makedirs(debugDir, exist_ok=True)

    # Step -1: padding
    h, w = image.shape[:2]
    padded = np.random.randint(
        z, t, (h + 2 * padSize, w + 2 * padSize, 3), dtype=np.uint8
    )
    padded[padSize : padSize + h, padSize : padSize + w] = image
    image = padded.copy()

    # Step 0: background masking
    grayMask = cv2.inRange(image, lowerGray, upperGray)
    nonBgMask = cv2.bitwise_not(grayMask)
    maskedImage = cv2.bitwise_and(image, image, mask=nonBgMask)

    cv2.imwrite(os.path.join(debugDir, "00_grayMask.png"), grayMask)
    cv2.imwrite(os.path.join(debugDir, "00_nonBgMask.png"), nonBgMask)

    # Step 1: edge detection
    gray = cv2.cvtColor(maskedImage, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    K = 15
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (K, K)) # shtupid code
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    cv2.imwrite(os.path.join(debugDir, "01_edges.png"), edges)
    cv2.imwrite(os.path.join(debugDir, "01b_closed.png"), closed)

    # Step 2: contour detection with hierarchy
    contours, hierarchy = cv2.findContours(
        closed.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    print(f"[INFO] Found {len(contours)} contours (with hierarchy)")

    # not a real step: but stores the top 10 contours                  V
    topContours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    debugTopContours = image.copy()

    for i, cnt in enumerate(topContours):
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(debugTopContours, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(
            debugTopContours,
            f"#{i} A={int(cv2.contourArea(cnt))}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2,
        )

    cv2.imwrite(os.path.join(debugDir, "03b_canidateContours.png"), debugTopContours)

    # Step 3: Filter contours (outermost only)
    filteredContours = []
    areaDebugInfo = []

    for cnt, h in zip(contours, hierarchy[0]):
        parent = h[3]  # -1 means no parent
        x, y, w, hBox = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        aspect = w / hBox if hBox != 0 else 0
        areaDebugInfo.append((area, aspect, parent, (x, y, w, hBox)))

        if parent == -1 and area > 50000 and 1.2 < aspect < 2.5:
            filteredContours.append(cnt)

    # Save debug area/aspect info
    areaDebugPath = os.path.join(debugDir, "02_contourAreas.txt")
    with open(areaDebugPath, "w") as f:
        for i, (area, aspect, parent, box) in enumerate(areaDebugInfo):
            f.write(
                f"Contour {i}: Area={area:.2f}, Aspect={aspect:.2f}, Parent={parent}, Box={box}\n"
            )

    # Step 4: Sort by area and keep top 6
    postcardContours = sorted(filteredContours, key=cv2.contourArea, reverse=True)[:6]
    print(f"[INFO] Filtered down to {len(postcardContours)} candidate contours")

    # Step 5: Debug visuals
    debugBoxes = image.copy()
    for i, cnt in enumerate(postcardContours):
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(debugBoxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            debugBoxes,
            str(i),
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

    cv2.imwrite(os.path.join(debugDir, "03_allBoxes.png"), debugBoxes)

    # Step 5.5: Save contour center coords
    contourCoords[baseName] = {}
    cardNumForFile = 0

    for cnt in postcardContours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        cardNumForFile += 1
        cardName = f"card{index + cardNumForFile - 1:04d}"
        contourCoords[baseName][cardName] = {"x": cX, "y": cY}

    # Step 6: Warp, rotate, and save
    savedCount = 0
    for cnt in postcardContours:
        rect = cv2.minAreaRect(cnt)
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

        if warped.shape[0] > warped.shape[1]:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

        outName = f"card{index:04d}_front.png"
        cv2.imwrite(os.path.join(outputDir, outName), warped)
        print(f"[SAVED] {outName}")
        index += 1
        savedCount += 1

    processedFiles.add(baseName)
    finalContoursDebug.append(f"{baseName}: {savedCount}")

# Final save
with open(seenPath, "w") as f:
    f.write("\n".join(sorted(processedFiles)))

with open(indexPath, "w") as f:
    f.write(str(index))

with open(contourDebugPath, "w") as f:
    f.write("\n".join(finalContoursDebug))

with open(contourCoordsPath, "w") as f:
    json.dump(contourCoords, f, indent=2)

print("\n[DONE] All front scans processed.")
