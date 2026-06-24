#fastapi
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import re
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

#initialize the FastAPI app
app = FastAPI(
    title="TEXT SUMMARIZER APP",
    description="This is a text summarizer app that uses the T5 model to summarize text.",
    version="1.0.0",
)

#model&tokenizer
MODEL_NAME = "./Text_Summarization"

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
#device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


model.to(device)

#template
templates = Jinja2Templates(directory=".")

#import model schema
class DialogueInput(BaseModel):
    dialogue: str
    
    
    
def clean_data(text):
    text = re.sub(r"\r\n", " ", text) #lines
    text = re.sub(r"\s+", " ", text) #extra spaces
    text = re.sub(r"<.*?>", " ", text) # html tags
    text = text.strip().lower() #strip and lower
    return text


def summarize_dialogue(dialogue : str) -> str:
    dialogue = clean_data(dialogue)

    inputs = tokenizer(
        dialogue,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.to(device)
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
#API endpoint
@app.post("/summarize/")
async def create_item(dialogue_input: DialogueInput):
    try:
        summary = summarize_dialogue(dialogue_input.dialogue)
        return {"summary": summary}
    except Exception as e:
        print("ERROR:", e)
        return {"summary": str(e)}

from fastapi.responses import FileResponse

@app.get("/")
async def home():
    return FileResponse("index.html")
