import os
import json
import torch
import numpy as np
import sounddevice as sd
import warnings
import re
import openvino as ov
import time

from transformers import AutoTokenizer, TextStreamer, AutoProcessor
from optimum.intel import OVModelForCausalLM, OVModelForSpeechSeq2Seq
from optimum.intel.openvino import OVModelForTextToSpeechSeq2Seq

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OV_NUM_STREAMS"] = "1"

warnings.filterwarnings("ignore", message="Defaulting repo_id to hexgrad/Kokoro-82M")

# ==========================================
# 1. HARDWARE DETECTION & SETUP
# ==========================================
print("🔍 Scanning hardware...")
core = ov.Core()
available_ov_devices = core.available_devices
print(f"   OpenVINO sees: {available_ov_devices}")

if "GPU" in available_ov_devices:
    ov_device = "GPU"
    print("   ✅ LLM will run on Intel iGPU!")
else:
    ov_device = "CPU"
    print("   ⚠️ OpenVINO GPU not found. LLM will run on CPU.")

# ==========================================
# 2. MODEL & HARDWARE INITIALIZATION
# ==========================================
print("\n🚀 Loading LLM...")
#llm_model_id = "OpenVINO/LFM2.5-350M-int8-ov"
#tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
llm_model_id = "./smollm2-135m-instruct-int8-ov"
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

llm_model = OVModelForCausalLM.from_pretrained(llm_model_id, device=ov_device, export=False)

print("🔊 Loading TTS (Kokoro via OpenVINO)...")
tts_model_id = "OpenVINO/Kokoro-82M-int8-ov"
tts_model = OVModelForTextToSpeechSeq2Seq.from_pretrained(tts_model_id, device="CPU", export=False)

print("🎤 Loading Whisper (Speech-to-Text via OpenVINO)...")
whisper_model_id = "OpenVINO/whisper-small.en-int8-ov"
whisper_processor = AutoProcessor.from_pretrained(whisper_model_id)
whisper_model = OVModelForSpeechSeq2Seq.from_pretrained(whisper_model_id, device="CPU", export=False)

# Global audio output stream
sample_rate = 24000
audio_stream = sd.RawOutputStream(samplerate=sample_rate, channels=1, dtype='float32', blocksize=2048)
audio_stream.start()

# ==========================================
# 3. TEXT CLEANING & CAPPING
# ==========================================
def clean_and_cap_for_tts(text, max_chars=400):
    if not text:
        return ""
    
    # 1. Remove any text inside parentheses (), brackets [], or curly braces {}
    text = re.sub(r'[\(\[\{].*?[\)\]\}]', '', text)
    
    # 2. Replace all types of dashes with a spaced hyphen
    text = text.replace('—', ' - ').replace('–', ' - ').replace('―', ' - ')
    
    # 3. Normalize ALL smart quotes and apostrophes to standard ASCII using explicit Unicode
    # \u2018 = left single quote, \u2019 = right single quote (the common smart apostrophe)
    # \u201c = left double quote, \u201d = right double quote
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    
    # 4. Keep ONLY: letters, numbers, spaces, and specific punctuation
    # Using a set makes this fast and perfectly clear
    allowed_chars = set(" .,!?'-")
    cleaned = ''.join(char for char in text if char.isalnum() or char in allowed_chars)
    
    # 5. Collapse extra spaces (crucial after removing bracketed text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    # 6. Safety truncation
    if len(cleaned) > max_chars:
        truncated = cleaned[:max_chars]
        last_punct = max(truncated.rfind('.'), truncated.rfind('!'), truncated.rfind('?'))
        if last_punct > max_chars // 2:
            cleaned = truncated[:last_punct + 1]
        else:
            cleaned = truncated.rstrip() + "..."
            
    return cleaned

# ==========================================
# 4. AUDIO RECORDING FUNCTION
# ==========================================
def record_audio():
    """Records audio until Enter is pressed, returns numpy array."""
    print("\n🎤 [Recording... Press Enter to stop]")
    
    # Record at 16kHz for Whisper
    record_sample_rate = 16000
    audio_chunks = []
    
    def audio_callback(indata, frames, time, status):
        if status:
            print(f"⚠️ {status}")
        audio_chunks.append(indata.copy())
    
    # Start recording
    with sd.InputStream(samplerate=record_sample_rate, channels=1, callback=audio_callback):
        input()  # Wait for Enter key
    
    # Combine chunks
    if not audio_chunks:
        return None
    
    audio_data = np.concatenate(audio_chunks, axis=0)
    return audio_data.flatten()

def transcribe_audio(audio_data):
    """Transcribes audio using Whisper."""
    if audio_data is None or len(audio_data) == 0:
        return ""
    
    try:
        # Process audio
        input_features = whisper_processor(
            audio_data,
            sampling_rate=16000,
            return_tensors="pt",
        ).input_features
        
        # Generate transcription
        outputs = whisper_model.generate(input_features)
        text = whisper_processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        return text.strip()
    except Exception as e:
        print(f"❌ [Whisper Error]: {str(e)}")
        return ""

# ==========================================
# 5. CHATBOT STATE & CONFIGURATION
# ==========================================
MAX_CONTEXT_TOKENS = 2048
# DEFAULT_SYSTEM = "You are a Sarah, a 27 year old female from Taiwan that loves anime and video games. CRITICAL: Keep responses to 1-2 short sentences maximum. Be warm and helpful. Never use markdown or formatting."
# 1. Short, positive, and direct system prompt
DEFAULT_SYSTEM = (
    "You are Sarah, a 27-year-old from Taiwan who loves anime and video games. "
    "Reply in exactly 1 or 2 short, friendly sentences. "
    "Speak naturally. Do not write lists, stories, or use markdown."
)

chat_history = [{"role": "system", "content": DEFAULT_SYSTEM}]

def get_token_count(history_list, tokenizer_obj):
    text_string = tokenizer_obj.apply_chat_template(history_list, tokenize=False)
    raw_ids = tokenizer_obj.encode(text_string, add_special_tokens=False)
    return len(raw_ids)

def speak_text(text, threshold=250):
    cleaned_text = clean_and_cap_for_tts(text)
    if not cleaned_text.strip():
        return

    sentences = re.split(r'(?<=[.!?]) +', cleaned_text)
    buffer = ""

    for sentence in sentences:
        if len(buffer) + len(sentence) < threshold:
            buffer += " " + sentence
        else:
            # synthesize next chunk
            inputs = tts_model.preprocess_input(buffer.strip(), voice="af_heart", lang_code="a")
            audio = tts_model.generate(**inputs)
            audio_np = audio.numpy().astype(np.float32)

            # wait before writing new audio
            sd.wait()
            audio_stream.write(audio_np.tobytes())

            buffer = sentence

    if buffer:
        inputs = tts_model.preprocess_input(buffer.strip(), voice="af_heart", lang_code="a")
        audio = tts_model.generate(**inputs)
        audio_np = audio.numpy().astype(np.float32)

        sd.wait()
        audio_stream.write(audio_np.tobytes())


# ==========================================
# 6. MAIN INTERACTIVE LOOP
# ==========================================
print("\n🧸 Chatbot Live! Preset Commands:")
print("  /system <text>  -> Change personality rules instantly")
print("  /reset          -> Clean chat memory (keeps current system instructions)")
print("  /status         -> Check active preset profile configurations")
print("  /save <name>    -> Save current system prompt as a named preset JSON")
print("  /load <name>    -> Load a saved system prompt preset instantly")
print("  /mute           -> Toggle TTS audio output on/off")
print("  /r              -> Record voice input (press Enter to stop)")
print("  quit            -> Close the application\n")

tts_enabled = True

while True:
    user_input = input("\nYou: ").strip()
    
    # Handle voice recording command
    if user_input.lower() == '/r':
        audio_data = record_audio()
        if audio_data is not None:
            print("🔄 [Transcribing...]")
            transcribed_text = transcribe_audio(audio_data)
            if transcribed_text:
                print(f"📝 [Transcribed: {transcribed_text}]")
                user_input = transcribed_text
            else:
                print("⚠️ [No speech detected]")
                continue
        else:
            print("⚠️ [Recording failed]")
            continue
    
    if user_input.lower() in ['quit', '/quit']:
        break

    if user_input.startswith('/system '):
        new_prompt = user_input[8:].strip()
        chat_history[0]["content"] = new_prompt
        print(f"⚙️ [System Prompt Updated]: '{new_prompt}'")
        continue

    if user_input.lower() == '/reset':
        chat_history = [{"role": "system", "content": chat_history[0]["content"]}]
        print("🧹 [Memory wiped clean! System preset retained.]")
        continue

    if user_input.lower() == '/status':
        current_tokens = get_token_count(chat_history, tokenizer)
        print("\n📊 --- PRESET CONFIG PANEL ---")
        print(f"🧠 Active Persona Prompt: \"{chat_history[0]['content']}\"")
        print(f"📟 Current State Buffer: {current_tokens} tokens active")
        print(f"🔊 TTS Audio Output: {'ENABLED' if tts_enabled else 'MUTED'}")
        print(f"⚙️ LLM Device: {ov_device} | TTS Device: CPU | Whisper Device: CPU")
        print("-------------------------------")
        continue

    if user_input.startswith('/save '):
        filename = user_input[6:].strip()
        if not filename.endswith('.json'):
            filename += '.json'
        os.makedirs("generated", exist_ok=True)
        filepath = os.path.join("generated", filename)
        export_preset = {"preset_name": filename.replace('.json', ''), "system_prompt": chat_history[0]["content"]}
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(export_preset, f, indent=4, ensure_ascii=False)
            print(f"💾 [Success]: System preset saved cleanly to '{filepath}'")
        except Exception as e:
            print(f"❌ [Error saving preset]: {str(e)}")
        continue

    if user_input.startswith('/load '):
        filename = user_input[6:].strip()
        if not filename.endswith('.json'):
            filename += '.json'
        filepath = os.path.join("generated", filename)
        if not os.path.exists(filepath):
            print(f"❌ [File Not Found]: Could not find a preset named '{filepath}'")
            continue
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                loaded_preset = json.load(f)
            chat_history = [{"role": "system", "content": loaded_preset["system_prompt"]}]
            print(f"📂 [Success]: Preset loaded from '{filepath}'")
        except Exception as e:
            print(f"❌ [Error loading preset]: {str(e)}")
        continue

    if user_input.lower() == '/mute':
        tts_enabled = not tts_enabled
        print(f"🔊 TTS Audio Output {'Muted' if not tts_enabled else 'Unmuted'}.")
        continue

    # --- Standard Chat Execution Block ---
    current_system_prompt = chat_history[0]["content"]
    chat_history = [
        {"role": "system", "content": current_system_prompt},
        {"role": "user", "content": user_input}
    ]

    model_inputs = tokenizer.apply_chat_template(
        chat_history,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    )

    input_ids = model_inputs["input_ids"].to(llm_model.device)

    # 2. Stricter generation parameters
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    print("\n🤖 Bot: ", end="", flush=True)

    # LFM 2.5
    output = llm_model.generate(
        input_ids=input_ids,
        do_sample=True,
        temperature=0.45,
        top_p=0.85,
        repetition_penalty=1.2,
        max_new_tokens=512,
        streamer=streamer
    ) 

    generated_tokens = output[0, input_ids.shape[-1]:]
    ai_response_string = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    print()

    if tts_enabled and ai_response_string:
        speak_text(ai_response_string)

    chat_history.append({"role": "assistant", "content": ai_response_string})

# ==========================================
# 7. CLEANUP
# ==========================================
print("\n👋 Shutting down...")
audio_stream.stop()
audio_stream.close()
print("✅ Goodbye!")
