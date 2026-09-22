from optimum.intel.openvino import OVModelForCausalLM

model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"

ov_model = OVModelForCausalLM.from_pretrained(
    model_id,
    export=True,
    dtype="int8",       # ✅ quantize to INT8
    compile=False,
    device="CPU"
)

ov_model.save_pretrained("./smollm2-135m-instruct-int8-ov")

