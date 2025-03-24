from evaluate import load

bleu = load("sacrebleu")
print(bleu.compute(predictions=["test summary"], references=[["test reference"]]))
