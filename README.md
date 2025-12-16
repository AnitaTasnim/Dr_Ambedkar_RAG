
# RAG Evaluation Quick Start

A quick way to evaluate Retrieval-Augmented Generation (RAG) pipelines with Mistral LLM.



## For detailed analysis of the results, see [results_analysis.md](results_analysis.md).




## 1. Setup

1. **Clone the repository:**

```bash
git clone https://github.com/sayedisaverygoodboy/assestment
cd assestment
````

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Create a `.env` file** in the project root:

```text
MISTRAL_API_KEY=your_api_key_here
```

This is required for `ChatMistralAI` to work. got to https://admin.mistral.ai/organization/api-keys , log in if not , then create an api key and copy paste in the env file

---


## 2. Configure Evaluation

Edit `evaluate.py` to adjust:

```python
CORPUS_DIR = "./corpus"
TEST_DATASET = "./test_dataset.json"
OUTPUT_DIR = "./results"
CHUNK_RANGES = [(200, 300), (500, 600), (800, 1000)]
OVERLAP = 0
TOP_K = 5
PERSIST_DIR = None
```

---

## 3. Run Evaluation

```bash
python evaluate.py
```

* Chunks will be created, embeddings built, retrieval performed, answers generated, and metrics computed.
* Results are saved in `./results/`.

---

## 4. Outputs

* `results_*.json`: per-chunk results
* `test_results.json`: combined results
* `results_analysis.md`: Markdown report with:

  * Metric tables
  * Plots
  * Failure mode analysis
  * Recommended chunk size

Plots are saved in `./results/plots/`.

---

## Notes

* Ensure your `.env` contains `MISTRAL_API_KEY`.
* BLEU, ROUGE-L, cosine similarity, relevance, and faithfulness metrics are computed automatically.
* You can switch to other LLMs by editing the `llm` initialization in `evaluate.py`.
 