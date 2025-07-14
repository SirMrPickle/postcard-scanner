import cv2
import numpy as np
import os
import glob
import re
import json

# ======= FRONT SCANNER (FORCED 6-CONTOUR MODE) ======= #

input_dir = "_INPUT"
output_dir = "output/front"
debug_base_dir = "debug/front"
seen_path = "counters/scanned_front.txt"
index_path = "counters/index_front.txt"
contour_debug = "debug/contours_front.txt"
contour_coords_path = "debug/front_coords.json"

log_debug = False
pad_size = 10  # noise padding around edges

z, t = 30, 55  # background gray range
lower_gray = np.array([z, z, z])
upper_gray = np.array([t, t, t])

os.makedirs(output_dir, exist_ok=True)
os.makedirs(debug_base_dir, exist_ok=True)
os.makedirs(os.path.dirname(seen_path), exist_ok=True)

# Load seen files
if os.path.exists(seen_path):
    with open(seen_path, "r") as f:
        processed_files = set(f.read().splitlines())
else:
    processed_files = set()

# Load index
if os.path.exists(index_path):
    with open(index_path, "r") as f:
        index = int(f.read().strip())
else:
    index = 1

# Load or initialize contour coordinates
if os.path.exists(contour_coords_path):
    with open(contour_coords_path, "r") as f:
        contour_coords = json.load(f)
else:
    contour_coords = {}


# Helper to extract scan number for proper order
def extract_number(filename):
    match = re.search(r"sc(\d+)[_-]front", filename.lower())
    return int(match.group(1)) if match else float("inf")


# Sort input files correctly
input_files = sorted(glob.glob(os.path.join(input_dir, "*.png")), key=extract_number)
final_contours_debug = []

for input_path in input_files:
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    if base_name in processed_files:
        print(f"[SKIP] {base_name} already processed")
        continue

    if "front" not in input_path.lower():
        continue

    print(f"\n[PROCESSING] {input_path}")
    image = cv2.imread(input_path)
    if image is None:
        print(f"[ERROR] Cannot open {input_path}, skipping.")
        continue

    debug_dir = os.path.join(debug_base_dir, base_name)
    os.makedirs(debug_dir, exist_ok=True)

    # Step -1: Padding
    h, w = image.shape[:2]
    padded = np.random.randint(
        z, t, (h + 2 * pad_size, w + 2 * pad_size, 3), dtype=np.uint8
    )
    padded[pad_size : pad_size + h, pad_size : pad_size + w] = image
    image = padded.copy()

    # Step 0: Background masking
    gray_mask = cv2.inRange(image, lower_gray, upper_gray)
    non_bg_mask = cv2.bitwise_not(gray_mask)
    masked_image = cv2.bitwise_and(image, image, mask=non_bg_mask)

    cv2.imwrite(os.path.join(debug_dir, "00_gray_mask.png"), gray_mask)
    cv2.imwrite(os.path.join(debug_dir, "00_non_bg_mask.png"), non_bg_mask)

    # Step 1: Edge detection
    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    cv2.imwrite(os.path.join(debug_dir, "01_edges.png"), edges)
    cv2.imwrite(os.path.join(debug_dir, "01b_closed.png"), closed)

    # Step 2: Contour detection
    contours, _ = cv2.findContours(
        closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    print(f"[INFO] Found {len(contours)} contours")

    # Step 3: Sort all contours by area, keep top 6
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
    postcard_contours = contours_sorted[:6]  # FORCE 6 even if some are bad                        <- FORCE AMOUNT OF CONTOURS

    # Step 4: Debug visuals
    debug_boxes = image.copy()
    for i, cnt in enumerate(postcard_contours):
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(debug_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            debug_boxes,
            str(i),
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

    cv2.imwrite(os.path.join(debug_dir, "03_all_boxes.png"), debug_boxes)

    # Step 4.5: Save contour center boxes in clean format
    contour_coords[base_name] = {}
    cardNumForFile = 0  # Track card number per input file

    for cnt in postcard_contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        cardNumForFile += 1
        cardName = f"card{index + cardNumForFile - 1:04d}"
        contour_coords[base_name][cardName] = {"x": cX, "y": cY}

    # Step 5: Warp, rotate, save
    saved_count = 0
    for cnt in postcard_contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect).astype(np.intp)
        width, height = int(rect[1][0]), int(rect[1][1])

        if width == 0 or height == 0:
            print("[WARN] Skipping contour with zero width/height")
            continue

        src_pts = box.astype("float32")
        dst_pts = np.array(
            [[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]],
            dtype="float32",
        )

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, M, (width, height))

        # Rotate to landscape
        if warped.shape[0] > warped.shape[1]:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

        out_name = f"card{index:04d}_front.png"
        cv2.imwrite(os.path.join(output_dir, out_name), warped)
        print(f"[SAVED] {out_name}")
        index += 1
        saved_count += 1

    processed_files.add(base_name)
    final_contours_debug.append(f"{base_name}: {saved_count}")

# Final save
with open(seen_path, "w") as f:
    f.write("\n".join(sorted(processed_files)))

with open(index_path, "w") as f:
    f.write(str(index))

with open(contour_debug, "w") as f:
    f.write("\n".join(final_contours_debug))

with open(contour_coords_path, "w") as f:
    json.dump(contour_coords, f, indent=2)

print("\n[DONE] All front scans processed.")
