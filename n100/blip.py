from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import pytesseract
import requests

# Load BLIP
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
url = "https://graphicdesigneye.com/assets/uploads/ckeditor/2025/Oct/silly-situations-funny-t-shirt-designs.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Caption with BLIP
inputs = processor(images=image, return_tensors="pt")
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)

# OCR with Tesseract
ocr_text = pytesseract.image_to_string(image)

print("Caption:", caption)
print("OCR Text:", ocr_text)
