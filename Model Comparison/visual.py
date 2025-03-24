import torch
import time
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import T5Tokenizer, T5ForConditionalGeneration
from evaluate import load


# Load tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Load evaluation metrics
rouge = load("rouge")
bleu = load("bleu")  # ✅ Use "sacrebleu" instead of "bleu"

# Load test dataset (Replace with actual test data)
test_articles = ["This is a sample article for testing.", "Another example article for model evaluation."]
test_summaries = ["Sample summary.", "Example summary."]

# Function to evaluate a model
def evaluate_model(model_name, model_path):
    model = T5ForConditionalGeneration.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    rouge_scores = []
    bleu_scores = []
    inference_times = []

    for i in range(len(test_articles)):  # Limit for faster evaluation
        input_text = "summarize: " + test_articles[i]
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(model.device)
        
        start_time = time.time()
        with torch.no_grad():
            output = model.generate(**inputs, max_length=150, num_beams=4, early_stopping=True)
        inference_times.append(time.time() - start_time)

        generated_summary = tokenizer.decode(output[0], skip_special_tokens=True)

        # Compute ROUGE
        rouge_result = rouge.compute(predictions=[generated_summary], references=[test_summaries[i]])
        rouge_scores.append(rouge_result["rougeL"])  # ✅ Fix: Directly append float score

        # Compute BLEU
        bleu_result = bleu.compute(predictions=[generated_summary], references=[[test_summaries[i]]])
        bleu_scores=[]
        bleu_scores.append(bleu_result["score"])  # ✅ Use "score" instead of "bleu"

    # Compute averages
    avg_rouge = sum(rouge_scores) / len(rouge_scores)
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    avg_time = sum(inference_times) / len(inference_times)

    return avg_rouge, avg_bleu, avg_time

# Evaluate both models
models = {
    "T5-Small (Pretrained)": "t5-small",
    "T5-Small (Fine-tuned)": r"C:\Users\lenovo\Desktop\Projects\Major Project\Fine-Tuned Model"
}

results = {}

for name, path in models.items():
    print(f"\nEvaluating {name}...")
    rouge_l, bleu, time_taken = evaluate_model(name, path)
    results[name] = {"ROUGE-L": rouge_l, "BLEU": bleu, "Inference Time": time_taken}

# Convert results to lists for plotting
metrics = ["ROUGE-L", "BLEU", "Inference Time"]
pretrained_values = [results["T5-Small (Pretrained)"][metric] for metric in metrics]
finetuned_values = [results["T5-Small (Fine-tuned)"][metric] for metric in metrics]

# Plot results
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(8, 5))
x_labels = ["ROUGE-L", "BLEU", "Inference Time (s)"]
bar_width = 0.4

x_indexes = range(len(metrics))
plt.bar(x_indexes, pretrained_values, width=bar_width, label="T5-Small (Pretrained)", alpha=0.7)
plt.bar([x + bar_width for x in x_indexes], finetuned_values, width=bar_width, label="T5-Small (Fine-tuned)", alpha=0.7)

plt.xticks([x + bar_width / 2 for x in x_indexes], x_labels)
plt.ylabel("Score / Time (s)")
plt.title("Model Comparison: T5-Small vs. Fine-tuned T5-Small")
plt.legend()
plt.show()
