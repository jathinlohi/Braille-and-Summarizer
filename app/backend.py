from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from newspaper import Article, Config
from transformers import T5ForConditionalGeneration, T5Tokenizer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import traceback

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom User-Agent for newspaper to avoid blocking
user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
news_config = Config()
news_config.browser_user_agent = user_agent
news_config.request_timeout = 10

# Load Summarization Model
try:
    model_path = r"C:\Users\lenovo\Desktop\Projects\Major Project\Fine-Tuned Model"
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path)
except Exception as e:
    print("Error loading T5 model:", str(e))
    model, tokenizer = None, None

# Load spaCy NER model
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    print("Error loading spaCy model:", str(e))
    nlp = None

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

def convert_to_braille(text: str) -> str:
    """Convert text to Braille safely."""
    try:
        return "".join(braille_dict.get(char.lower(), char) for char in text)
    except Exception as e:
        print("Error in Braille conversion:", str(e))
        return "Braille conversion failed."

def generate_summary(text: str) -> str:
    """Generate a summary using the T5 model."""
    if model is None or tokenizer is None:
        return "Summarization model not available."
    try:
        inputs = tokenizer.encode("summary: " + text, return_tensors="pt", max_length=1024, truncation=True)
        outputs = model.generate(inputs, max_length=250, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary if summary else "Summary generation failed."
    except Exception as e:
        print("Error generating summary:", str(e))
        traceback.print_exc()
        return "Summary generation failed."

def extract_key_entity(text: str) -> str:
    """Extract key entity or keyword from text."""
    if nlp is None:
        return "NER model unavailable."
    try:
        doc = nlp(text)
        persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        if persons:
            return persons[0]

        # Fallback: Use TF-IDF keywords
        vectorizer = TfidfVectorizer(stop_words="english", max_features=5)
        tfidf_matrix = vectorizer.fit_transform([text])
        keywords = vectorizer.get_feature_names_out()
        return keywords[0] if keywords else "Unknown"
    except Exception as e:
        print("Error extracting entity:", str(e))
        return "Entity extraction failed."

@app.post("/summarize")
async def summarize_article(request: URLRequest):
    """Fetch, summarize, and extract data from article."""
    try:
        article = Article(request.url, config=news_config)
        article.download()
        article.parse()

        if not article.text.strip():
            raise HTTPException(status_code=400, detail="Failed to extract article text.")

        summary = generate_summary(article.text)
        key_entity = extract_key_entity(article.text)

        formatted_summary = f"{article.title}\n\n{summary}"
        braille_summary = convert_to_braille(formatted_summary)

        return {
            "title": article.title or "No Title",
            "summary": formatted_summary,
            "braille_summary": braille_summary,
            "key_entity": key_entity
        }

    except HTTPException as http_err:
        raise http_err

    except Exception as e:
        print("Error processing article:", str(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="An error occurred while processing the article.")
