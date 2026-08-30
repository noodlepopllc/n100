import openvino as ov
import numpy as np
from PIL import Image
import huggingface_hub as hf_hub
import os

def init():
    # 1. Download the model (runs once)
    model_id = "ibrhr/Real-ESRGAN-OpenVINO"
    model_dir = "Real-ESRGAN-OpenVINO"
    if not os.path.exists(model_dir):
        print("Downloading Real-ESRGAN OpenVINO model...")
        hf_hub.snapshot_download(model_id, local_dir=model_dir)

    # 2. Load and compile the model on GPU
    core = ov.Core()

    # Pick your model:
    # "real_esrgan_x2plus_fp32"          -> 2x general
    # "real_esrgan_x4plus_fp32"          -> 4x general  
    # "real_esrgan_x4plus_anime_6B_fp32" -> 4x anime (best for stylized art!)
    model_name = "real_esrgan_x4plus_anime_6B_fp32"

    xml_path = os.path.join(model_dir, "models", f"{model_name}.xml")
    bin_path = os.path.join(model_dir, "models", f"{model_name}.bin")

    print(f"Loading {model_name} on CPU...")
    sr_model = core.compile_model(xml_path, "CPU")
    return sr_model

def upscale_image_esrgan(pil_image):
    # Ensure RGB
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    # Input: raw uint8 NHWC
    img_np = np.array(pil_image).astype(np.uint8)
    img_np = img_np[np.newaxis, ...]  # NHWC + batch

    # Inference
    sr_model = init()
    input_layer = sr_model.input(0)
    output_layer = sr_model.output(0)
    outputs = sr_model.infer_new_request({input_layer: img_np})
    result = outputs[output_layer][0]  # remove batch

    # Convert NCHW → NHWC
    if result.shape[0] == 3:
        result = np.transpose(result, (1, 2, 0))

    # Output is float32 [0,1] → rescale to [0,255]
    result = np.clip(result * 255.0, 0, 255).astype(np.uint8)

    return Image.fromarray(result)

def main():
    import sys
    image = Image.open(sys.argv[1])
    upscale_image_esrgan(image).save(sys.argv[1].replace('.png','_upscaled.png'))

if __name__ == '__main__':
    main()

