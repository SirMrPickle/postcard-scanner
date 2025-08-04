import requests
from PIL import Image

API_URL = "https://router.huggingface.co/v1/chat/completions"
with open("HF_TOKEN", "r") as f: HF_TOKEN = f.read()

img = Image.open("your_image.png")

prompt = "Describe the image"
payload = img

print(requests.post(API_URL, headers={"Authorization": "Bearer YOUR_HUGGINGFACE_TOKEN"}, json={"inputs": prompt and payload}).json())