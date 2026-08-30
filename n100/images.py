import openvino_genai as ov_genai
from PIL import Image
import huggingface_hub as hf_hub
import time

model_id = "OpenVINO/LCM_Dreamshaper_v7-int8-ov"
model_path = "LCM_Dreamshaper_v7-int8-ov"


hf_hub.snapshot_download(model_id, local_dir=model_path)

device = "GPU"
pipe = ov_genai.Text2ImagePipeline(model_path, device)

# Track timing with Python's time module
step_start_time = None
step_times = []

def step_callback(step, num_inference_steps, timestamp):
    global step_start_time
    
    current_time = time.time()
    
    # Calculate time for this step
    if step_start_time is not None:
        step_duration = current_time - step_start_time
        step_times.append(step_duration)
        progress = ((step + 1) / num_inference_steps) * 100
        print(f"Step {step + 1}/{num_inference_steps} ({progress:.1f}%) - Time: {step_duration:.3f}s")
    else:
        print(f"Step 1/{num_inference_steps} (12.5%) - Starting...")
    
    step_start_time = current_time

prompt = "a beautiful woman in a swimsuit at the pool"

start_time = time.time()
image_tensor = pipe.generate(
    prompt, 
    num_inference_steps=10,
    width=512,
    height=512,
    callback=step_callback
)
total_time = time.time() - start_time

print(f"\n✅ Total generation time: {total_time:.3f}s")
if len(step_times) > 0:
    print(f"Average time per step: {sum(step_times)/len(step_times):.3f}s")

image = Image.fromarray(image_tensor.data[0])
image.save('test_int8_dream.png')
