from fastapi import FastAPI, File, UploadFile
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
from typing import Annotated
import soundfile as sf
import io

app = FastAPI()

# Initialize the translation model
model = T5ForConditionalGeneration.from_pretrained("t5-base")
tokenizer = T5Tokenizer.from_pretrained("t5-base")

# Initialize the speech recognition model
model_id = "openai/whisper-tiny"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float32

model1 = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, use_safetensors=True
)
model1.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model1,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

@app.get("/")
def home():
    return {"message": "Hello world"}


@app.post("/translate")
def translate(data: dict):
    translated_text = translate_text(data["inputText"], data["sourceLang"], data["targetLang"])
    return {"translatedText": translated_text}

def translate_text(inputText: str, sourceLang: str, targetLang: str) -> str:
    prefix = f"translate {sourceLang} to {targetLang}: "
    inputText = prefix + inputText
    input_ids = tokenizer(inputText, return_tensors="pt").input_ids
    translated_ids = model.generate(input_ids)
    translatedText = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
    return translatedText

@app.post("/speech-recognition")
async def speech_recognition(audio: UploadFile ):
    audio_data = await audio.read()  # Read the file data
    data, samplerate = sf.read(io.BytesIO(audio_data)) 
    result = pipe(data)
    return {"inputText": result["text"]}