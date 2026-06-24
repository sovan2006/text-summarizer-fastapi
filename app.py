from fastapi import FastAPI
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
from fastapi.responses import FileResponse
import torch
import re
import traceback

# =====================================
# FastAPI App
# =====================================

app = FastAPI(
    title="TEXT SUMMARIZER APP",
    description="Dialogue Summarization using Fine-Tuned T5 Model",
    version="1.0.0"
)

# =====================================
# Load Model
# =====================================

MODEL_NAME = "./Text_Summarization"

print("Loading tokenizer...")
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# =====================================
# Device
# =====================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"Model loaded on {device}")

# =====================================
# Input Schema
# =====================================

class DialogueInput(BaseModel):
    dialogue: str

# =====================================
# Text Cleaning
# =====================================

def clean_data(text):
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.strip().lower()
    return text

# =====================================
# Summarization Function
# =====================================

def summarize_dialogue(dialogue):

    dialogue = clean_data(dialogue)

    input_text = "summarize: " + dialogue

    inputs = tokenizer(
        input_text,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

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

# =====================================
# Home Page
# =====================================

@app.get("/")
async def home():
    return FileResponse("index.html")

# =====================================
# Health Check
# =====================================

@app.get("/health")
async def health():
    return {
        "status": "running",
        "device": str(device)
    }

# =====================================
# Test Model Endpoint
# =====================================

@app.get("/test-model")
async def test_model():
    try:
        text = "summarize: John and Sarah have a meeting tomorrow."

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=50
            )

        summary = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return {"summary": summary}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

# =====================================
# Summarization API
# =====================================

@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):

    try:

        summary = summarize_dialogue(
            dialogue_input.dialogue
        )

        return {
            "summary": summary
        }

    except Exception as e:

        traceback.print_exc()

        return {
            "error": str(e)
        }