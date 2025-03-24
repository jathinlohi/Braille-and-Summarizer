from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from newspaper import Article
from transformers import T5ForConditionalGeneration, T5Tokenizer

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (restrict in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Load Summarization Model
model_path = r"C:\Users\lenovo\Desktop\Projects\Major Project\Fine-Tuned Model"
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)

# Braille Mapping
braille_dict = {
    "a": "⠁", "b": "⠃", "c": "⠉", "d": "⠙", "e": "⠑",
    "f": "⠋", "g": "⠛", "h": "⠓", "i": "⠊", "j": "⠚",
    "k": "⠅", "l": "⠇", "m": "⠍", "n": "⠝", "o": "⠕",
    "p": "⠏", "q": "⠟", "r": "⠗", "s": "⠎", "t": "⠞",
    "u": "⠥", "v": "⠧", "w": "⠺", "x": "⠭", "y": "⠽", "z": "⠵",
    "0": "⠼⠚", "1": "⠼⠁", "2": "⠼⠃", "3": "⠼⠉", "4": "⠼⠙",
    "5": "⠼⠑", "6": "⠼⠋", "7": "⠼⠛", "8": "⠼⠓", "9": "⠼⠊",
    " ": " ", ",": "⠂", ".": "⠄", "?": "⠦", "!": "⠖",
    "-": "⠤", "'": "⠄", ":": "⠒", ";": "⠆", "(": "⠶", ")": "⠶"
}

class URLRequest(BaseModel):
    url: str

def convert_to_braille(text):
    return "".join(braille_dict.get(char.lower(), char) for char in text)

def generate_summary(text):
    inputs = tokenizer.encode("summary: " + text, return_tensors="pt", max_length=1024, truncation=True)
    outputs = model.generate(inputs, max_length=300, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

@app.post("/summarize")
async def summarize_article(request: URLRequest):
    try:
        article = Article(request.url)
        article.download()
        article.parse()
        if not article.text:
            raise HTTPException(status_code=400, detail="Failed to extract article text.")
        
        summary = generate_summary(article.text)
        formatted_summary = f"{article.title}\n\n{summary}"
        braille_summary = convert_to_braille(formatted_summary)
        return {"title": article.title, "summary": formatted_summary, "braille_summary": braille_summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))