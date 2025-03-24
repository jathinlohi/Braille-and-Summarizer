import tkinter as tk
from tkinter import ttk, messagebox
from transformers import BartForConditionalGeneration, BartTokenizer
from newspaper import Article

# Unicode Braille Dictionary
braille_dict = {
    # alphabet
    "a": "⠁", "b": "⠃", "c": "⠉", "d": "⠙", "e": "⠑",
    "f": "⠋", "g": "⠛", "h": "⠓", "i": "⠊", "j": "⠚",
    "k": "⠅", "l": "⠇", "m": "⠍", "n": "⠝", "o": "⠕",
    "p": "⠏", "q": "⠟", "r": "⠗", "s": "⠎", "t": "⠞",
    "u": "⠥", "v": "⠧", "w": "⠺", "x": "⠭", "y": "⠽", "z": "⠵",

    # number
    "0": "⠼⠚", "1": "⠼⠁", "2": "⠼⠃", "3": "⠼⠉", "4": "⠼⠙",
    "5": "⠼⠑", "6": "⠼⠋", "7": "⠼⠛", "8": "⠼⠓", "9": "⠼⠊",

    # punctuation
    " ": " ", ",": "⠂", ".": "⠄", "?": "⠦", "!": "⠖", "-": "⠤",
    "'": "⠄", ":": "⠒", ";": "⠆", "(": "⠶", ")": "⠶"
}

# Load the summarization model
model_name = "facebook/bart-large-cnn"
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

def extract_article(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        if not article.text:
            raise ValueError("No article text extracted.")
        return article.title, article.text
    except Exception as e:
        messagebox.showerror("Extraction Error", f"Error extracting article content: {e}")
        return None, None

def generate_summary(text):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    outputs = model.generate(inputs, max_length=300, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

def convert_to_braille(text):
    braille_text = []
    for char in text:
        if char.isupper():
            braille_text.append("⠠")  # Braille uppercase prefix
        braille_text.append(braille_dict.get(char.lower(), char))  # Convert lowercase and symbols
    return "".join(braille_text)

def summarize_only():
    url = url_entry.get().strip()
    if not url:
        messagebox.showerror("Input Error", "Please enter a valid article URL.")
        return

    headline, article_text = extract_article(url)
    if not article_text:
        return

    summary = generate_summary(article_text)
    output_text.delete(1.0, tk.END)
    output_text.insert(tk.END, f"{headline}\n\n{summary}")

def summarize_and_convert():
    url = url_entry.get().strip()
    if not url:
        messagebox.showerror("Input Error", "Please enter a valid article URL.")
        return

    headline, article_text = extract_article(url)
    if not article_text:
        return

    summary = generate_summary(article_text)
    braille_summary = convert_to_braille(summary)
    output_text.delete(1.0, tk.END)
    output_text.insert(tk.END, braille_summary)

root = tk.Tk()
root.title("Article Summarizer & Braille Converter")
root.geometry("800x700")

frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky="nsew")

url_label = ttk.Label(frame, text="Article URL:")
url_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)
url_entry = ttk.Entry(frame, width=80)
url_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

summarize_button = ttk.Button(frame, text="Summarize", command=summarize_only)
summarize_button.grid(row=1, column=0, padx=5, pady=10)

braille_button = ttk.Button(frame, text="Summarize & Convert to Braille", command=summarize_and_convert)
braille_button.grid(row=1, column=1, padx=5, pady=10)

output_label = ttk.Label(root, text="Summary Output:")
output_label.grid(row=1, column=0, sticky="w", padx=10, pady=5)

output_text = tk.Text(root, wrap="word", height=30, width=80)
output_text.grid(row=2, column=0, padx=10, pady=5, sticky="nsew")

root.grid_rowconfigure(2, weight=1)
root.grid_columnconfigure(0, weight=1)

root.mainloop()
