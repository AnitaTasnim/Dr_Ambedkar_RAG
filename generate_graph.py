import json
import matplotlib.pyplot as plt

# Load evaluation results
with open("test_results.json", "r", encoding="utf-8") as f:
    results = json.load(f)

chunk_sizes = [r["chunk_size"] for r in results]

# Metrics to plot
metrics = ["hit_rate", "mrr", "precision@3", "rougeL", "bleu", "cosine"]

plt.figure(figsize=(12, 8))

for metric in metrics:
    values = [r[metric] for r in results]
    plt.plot(chunk_sizes, values, marker='o', label=metric)

plt.title("Evaluation Metrics vs. Chunk Size")
plt.xlabel("Chunk Size (characters)")
plt.ylabel("Score")
plt.xticks(chunk_sizes)
plt.grid(True)
plt.legend()
plt.show()
