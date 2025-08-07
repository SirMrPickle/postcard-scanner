import os
import ollama
import re
import json
import time

# ========================================INFO=======================================
# 99% of the coding up until this point has been happening on my Mac laptop.        |
# All of this code is built for my windows machine, because of its processing power.|
# The following script locally loads the entire model, and runs it locally.         |
# ========================================INFO=======================================


model = "gemma3:4b"

# Set up the images
imageFolder = open("SENSITIVE/IMAGE_FOLDER", "r").read()
images = [img for img in os.listdir(imageFolder) if img.lower().endswith(".png")]

# Parse out and clean up JSON
def cleanJSON(content, isRaw=True):
    if isRaw: # this was tedious
        cleaned = re.sub(r"^```json\s*|```$", "", content.strip(), flags=re.IGNORECASE | re.MULTILINE).strip()
        parsed = json.loads(cleaned)
        return cleanJSON(parsed, isRaw=False)
    if isinstance(content, dict):
        return {k: cleanJSON(v, isRaw=False) for k, v in content.items()}
    if isinstance(content, list):
        return [cleanJSON(elem, isRaw=False) for elem in content]
    if isinstance(content, str) and content.strip().lower() in ("unknown", "not avalible"):
        return ""
    return content


# I'm very proud of these, they work really well.
prompt = """This is a vintage postcard. Carefully analyze the image in much detail, and prepare to export the found data into a JSON structure. 
Use all of the present text on the image to your advantage. Please do not generate any text content that cannot be found in the photo, for example, sender or reciever details, and printed or handwritten text.
Only respond with a JSON structure, and no plain text. If there is no title present, create a fitting title, with no more than 5 words. 
If data for a field cannot be found, do not insert Unknown, instead please leave it empty. Do not include escape characters in your response. Assume the longitude and latitude to the best of your ability.
Please format your response in the following JSON structure."""
jsonStructure = """{
        "title": "",
        "description": "",
        "estimated_date": "",
        "location_depicted": {
            "street_address": "",
            "city": "",
            "state": "",
            "country": "",
            "longitude": "",
            "latitude": "",
        },
        "front": {
            "caption": "",
            "image_type": "Photograph/Cartoon/Illustration",
            "color": "Full Color/B&W/Sepia",
            "publisher": "",
            "series_number": ""
        },
        "back": {
            "printed_text": "",
            "handwritten_text": "",
            "legibility": "Excelent/Moderate/Poor/Illegible",
            "language": "",
            "text_style": ""
        },
        "sender": {
            "name": "",
            "city": "",
            "state": "",
            "country": "",
            "address": "",
            "date_sent": ""
        },
        "recipient": {
            "name": "",
            "address": "",
            "date_received": ""
        },
        "condition": {
            "rating": "Mint/Good/Fair/Poor",
            "damage": [
                ""
            ],
            "damage_notes": ""
        },
        "general_notes": ""
    }"""

# load JSON file
if os.path.exists("analysis.json"):
    with open("analysis.json", "r") as f:
        allData = json.load(f)
else:
    allData = {}

# Process each image
for imageName in images:
    startTime = time.time()
    imagePath = os.path.join(imageFolder, imageName)

    jsonKeyName = imageName # This already has a .png extension as the name

    if jsonKeyName in allData:
        print(f"Skipping {jsonKeyName} (already processed).")
        continue

    print(f"Processing {imageName}...")

    with open(imagePath, "rb") as imageFile:
        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt + jsonStructure,
                    "images": [imageFile.read()],
                }
            ],
        )
    content = response["message"]["content"]

    clean = cleanJSON(content)
    allData[jsonKeyName] = clean
    with open("analysis.json", "w") as f:
        json.dump(dict(sorted(allData.items())), f, indent=4)
    print(f"Saved {jsonKeyName} > [{time.time()-startTime:.3}s @ {time.strftime('%H:%M:%S')}]")
    # Saved card0135.png > [10.0s @ 13:22:54]