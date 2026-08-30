from transformers import AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM

model_id = "OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int8-ov"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = OVModelForCausalLM.from_pretrained(model_id, device="GPU")

inputs = tokenizer("write a quick sort algorithm in C 89/90", return_tensors="pt")

outputs = model.generate(**inputs, max_length=1024)
text = tokenizer.batch_decode(outputs)[0]
print(text)

