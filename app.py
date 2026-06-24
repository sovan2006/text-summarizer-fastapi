from fastapi import FastAPI
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
from fastapi.responses import FileResponse
import torch
import re

# Initialize FastAPI app
app = FastAPI(
    title="TEXT SUMMARIZER APP",
    description="Text Summarization using T5 Model",
    version="1.0.0",
)

# Load model from local folder
MODEL_NAME = "./Text_Summarization"

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# Device setup
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)

# Input schema
class DialogueInput(BaseModel):
    dialogue: str


# Clean text
def clean_data(text):
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.strip().lower()
    return text


# Summarization function
def summarize_dialogue(dialogue: str) -> str:
    dialogue = clean_data(dialogue)

    inputs = tokenizer(
        dialogue,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.eval()

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=128,
            num_beams=4,
            early_stopping=True
        )

    summary = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    )

    return summary


# Home Page
@app.get("/")
async def home():
    return FileResponse("index.html")


# Summarization API
@app.post("/summarize/")
async def create_item(dialogue_input: DialogueInput):
    try:
        summary = summarize_dialogue(dialogue_input.dialogue)
        return {"summary": summary}
    except Exception as e:
    import traceback
    traceback.print_exc()
    return {"error": str(e)}


# Health Check
@app.get("/health")
async def health():
    return {"status": "running"}