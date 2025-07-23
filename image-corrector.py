import os
import sys
from PIL import Image

# POC right now, I don't know how much use this is going to serve me.
# Some are rotated correctly off the bat, byt others arent. It isnt all standard
showSkip = False
rotateIDs = ["100", "30"]

def rotateToPortrait(imagePath):
    try:
        with Image.open(imagePath) as img:
            width, height = img.size
            if width > height:
                print(f"[ROTATE] {imagePath} ({width}x{height})")
                img = img.rotate(90, expand=True)
                img.save(imagePath)
            elif showSkip:
                print(f"[SKIP] {imagePath} already portrait")
    except Exception as e:
        print(f"[ERROR] Failed on {imagePath}: {e}")

def processPath(path):
    if os.path.isfile(path):
        if path.lower().endswith(".png") and path.lower().startswith("sc" + "somehow the ID here"):
            rotateToPortrait(path)
    elif os.path.isdir(path):
        for root, _, files in os.walk(path):
            for name in files:
                filePath = os.path.join(root, name)
                if filePath.lower().endswith(".png"):
                    rotateToPortrait(filePath)
    else:
        print(f"[ERROR] Path not found: {path}")
        
processPath("_RAW")