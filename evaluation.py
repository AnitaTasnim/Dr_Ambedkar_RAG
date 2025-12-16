from dotenv import load_dotenv
import argparse
import json
import os
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
import statistics

# NLP / embedding / retrieval / llm
# using mistral
from langchain_mistralai import ChatMistralAI




from sentence_transformers import SentenceTransformer
import nltk

# plotting
import matplotlib.pyplot as plt

# internal imports
from evaluation_functions import (retrieval_hit_rate, retrieval_mrr, precision_at_k,
                                  compute_rouge_l, compute_bleu, compute_cosine_sim,
                                  compute_relevance, compute_faithfulness)
from utils import ensure_dir, prepare_corpus, smart_chunk_text ,index_chunks

# load environment variables from .env file
load_dotenv()



nltk.download('punkt', quiet=True)
nltk.download('punkt_tab')





# -------------------- LLM + Answer Generation --------------------
def generate_answer(llm: ChatMistralAI, question: str, retrieved_texts: List[str]) -> Tuple[str, List[str]]:
    """
    Return (answer, extracted_sources) using Groq LLM.
    """
    prompt_template = """You are an assistant that has access to retrieved passages. Answer the question concisely in one paragraph.
    Then on a new line write: SOURCES: <comma-separated list of source filenames used>

    Question: {question}

    Passages:
    {passages}

    Answer:
    """
    formatted = prompt_template.format(question=question, passages="\n\n".join(retrieved_texts))

    # Groq uses messages: list of tuples (role, content)
    messages = [
        ("system", "You are a helpful assistant."),
        ("user", formatted)
    ]

    resp = llm.invoke(messages)
    answer_text = resp.content.strip()

    # extract SOURCES
    sources = []
    for line in answer_text.splitlines()[::-1]:
        if line.strip().lower().startswith("sources:"):
            src_line = line.split(":", 1)[1]
            pieces = re.split(r"[,;]+", src_line)
            sources = [s.strip() for s in pieces if s.strip()]
            break

    return answer_text, sources

# -------------------- Analysis --------------------
def analyze_results(all_results: Dict[str, Any], output_dir: str, top_n_failures: int = 5) -> None:
    """
    Create a human-readable analysis file (results_analysis.md) and plots.
    """
    ensure_dir(output_dir)
    md_lines = []
    md_lines.append(f"# RAG Evaluation Analysis\n")
    md_lines.append(f"Generated: {datetime.utcnow().isoformat()} UTC\n")
    md_lines.append("\n## Aggregate comparison\n")

    # For plots: collect metric means per chunk key
    metric_keys = ['hit_rate', 'mrr', 'precision@1', 'precision@5', 'rouge_l', 'bleu', 'cosine_sim', 'relevance', 'faithfulness']
    chunk_keys = list(all_results.keys())
    # gather means
    means = {m: [] for m in metric_keys}
    for ck in chunk_keys:
        agg = all_results[ck]['aggregate']
        for m in metric_keys:
            means[m].append(agg.get(m, {}).get('mean', 0.0))

    # write table
    header = "| Chunk Size | " + " | ".join([k.replace('_', ' ').upper() for k in metric_keys]) + " |"
    sep = "|---" * (len(metric_keys) + 1) + "|"
    md_lines.append(header)
    md_lines.append(sep)
    for i, ck in enumerate(chunk_keys):
        row = f"| {ck} | " + " | ".join(f"{means[m][i]:.3f}" for m in metric_keys) + " |"
        md_lines.append(row)

    # plots - one plot per metric
    plots_dir = Path(output_dir) / "plots"
    ensure_dir(str(plots_dir))
    for m in metric_keys:
        plt.figure(figsize=(8, 4))
        plt.plot(chunk_keys, means[m], marker='o')
        plt.title(m.replace('_', ' ').title())
        plt.xlabel("Chunk Size")
        plt.ylabel("Mean")
        plt.grid(True)
        plt.tight_layout()
        fname = plots_dir / f"{m}.png"
        plt.savefig(fname)
        plt.close()
        md_lines.append(f"\n![{m}](plots/{fname.name})\n")

    # failure mode extraction: worst per-chunk by rouge_l (or hit_rate)
    md_lines.append("\n## Failure Mode Analysis\n")
    for ck in chunk_keys:
        per_q = all_results[ck]['per_question']
        # sort by rouge_l ascending (worst first) for answerable questions
        worst = sorted([q for q in per_q if q.get('answerable', True)], key=lambda x: x['metrics'].get('rouge_l', 0.0))
        md_lines.append(f"\n### Chunk: {ck} — Worst {top_n_failures} by ROUGE-L\n")
        for w in worst[:top_n_failures]:
            md_lines.append(f"- ID: {w['id']} — Q: {w['question'][:120]}...\n")
            md_lines.append(f"  - ROUGE-L: {w['metrics']['rouge_l']:.3f}, BLEU: {w['metrics']['bleu']:.3f}, CosSim: {w['metrics']['cosine_sim']:.3f}\n")
            md_lines.append(f"  - Retrieved: {w['retrieved_sources']}\n")
            short_ans = (w['answer'][:250] + '...') if len(w['answer']) > 250 else w['answer']
            md_lines.append(f"  - Answer (truncated): `{short_ans}`\n")

    # recommendation heuristics: simple rule-based
    # choose chunk with highest mean(relevance + faithfulness + hit_rate)
    scores = {}
    for ck in chunk_keys:
        agg = all_results[ck]['aggregate']
        score = (agg['relevance']['mean'] + agg['faithfulness']['mean'] + agg['hit_rate']['mean']) / 3.0
        scores[ck] = score
    best_ck = max(scores.items(), key=lambda x: x[1])[0]
    md_lines.append(f"\n## Recommendation\nBased on a combined heuristic (relevance + faithfulness + hit_rate) the recommended chunk configuration is **{best_ck}**.\n")

    # save analysis markdown
    analysis_path = Path(output_dir) / "results_analysis.md"
    analysis_path.write_text("\n".join(md_lines), encoding='utf-8')
    print(f"Saved analysis to {analysis_path}")


# -------------------- Main evaluation driver --------------------
def evaluate_all(corpus_dir: str, test_dataset_path: str, output_dir: str,
                 chunk_ranges: List[Tuple[int, int]], overlap: int = 0, top_k: int = 5,
                 embeddings_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 persist_directory: str = None):
    ensure_dir(output_dir)
    texts = prepare_corpus(corpus_dir)
    with open(test_dataset_path, 'r', encoding='utf-8') as f:
        test = json.load(f)
    questions = test['test_questions']

    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    # llm = Ollama(model="mistral-7b", verbose=False)
    # mistral-7b is not avilable in mistral's cloud  
    llm = ChatMistralAI(model_name="mistral-large-2512", temperature=0.0)
    

    all_results = {}

    for (min_c, max_c) in chunk_ranges:
        key = f"{min_c}_{max_c}"
        print(f"\n--- Evaluating chunks {min_c}-{max_c} (overlap={overlap}) ---")
        # build chunks
        chunks = []
        cid = 0
        for fname, text in texts.items():
            pieces = smart_chunk_text(text, min_c, max_c, overlap=overlap)
            for ptxt in pieces:
                chunks.append({'id': f"{fname}#c{cid}", 'text': ptxt, 'source': fname})
                cid += 1

        vectordb = index_chunks(chunks, embeddings_model_name=embeddings_model_name, persist_directory=persist_directory)

        per_q_results = []
        for q in questions:
            qid = q['id']
            qtext = q['question']
            ground = q['ground_truth']
            gt_docs = q.get('source_documents', [])

            docs = vectordb.similarity_search_with_score(qtext, k=top_k)
            retrieved_texts = [d.page_content for d, score in docs]
            retrieved_meta = [d.metadata.get('source') for d, score in docs]

            # retrieval metrics
            hit = retrieval_hit_rate(retrieved_meta, gt_docs)
            mrr = retrieval_mrr(retrieved_meta, gt_docs)
            p1 = precision_at_k(retrieved_meta, gt_docs, 1)
            p5 = precision_at_k(retrieved_meta, gt_docs, top_k)

            # LLM answer + extracted sources
            answer, extracted_sources = generate_answer(llm, qtext, retrieved_texts[:3])

            # answer metrics
            rouge_l = compute_rouge_l(answer, ground) if q.get('answerable', True) else 0.0
            bleu = compute_bleu(answer, ground) if q.get('answerable', True) else 0.0
            cos_sim = compute_cosine_sim(st_model, answer, ground) if q.get('answerable', True) else 0.0
            relevance = compute_relevance(answer, ground) if q.get('answerable', True) else 0.0
            faith = compute_faithfulness(answer, " ".join(retrieved_texts))

            per_q_results.append({
                'id': qid,
                'question': qtext,
                'retrieved_sources': retrieved_meta,
                'extracted_sources': extracted_sources,
                'answer': answer,
                'metrics': {
                    'hit_rate': hit,
                    'mrr': mrr,
                    'precision@1': p1,
                    f'precision@{top_k}': p5,
                    'rouge_l': rouge_l,
                    'bleu': bleu,
                    'cosine_sim': cos_sim,
                    'relevance': relevance,
                    'faithfulness': faith
                },
                'answerable': q.get('answerable', True)
            })

        # aggregate
        metric_names = ['hit_rate', 'mrr', 'precision@1', f'precision@{top_k}', 'rouge_l', 'bleu', 'cosine_sim', 'relevance', 'faithfulness']
        agg = {}
        for m in metric_names:
            vals = [q['metrics'].get(m, 0.0) for q in per_q_results]
            if vals:
                agg[m] = {'mean': statistics.mean(vals), 'median': statistics.median(vals), 'stdev': statistics.pstdev(vals) if len(vals) > 1 else 0.0}
            else:
                agg[m] = {'mean': 0.0, 'median': 0.0, 'stdev': 0.0}

        all_results[key] = {'per_question': per_q_results, 'aggregate': agg}

        # save per-chunk intermediate
        out_path = Path(output_dir) / f"results_{key}.json"
        out_path.write_text(json.dumps(all_results[key], indent=2), encoding='utf-8')
        print(f"Saved intermediate results to {out_path}")

    # final save
    final_path = Path(output_dir) / "test_results.json"
    final_path.write_text(json.dumps(all_results, indent=2), encoding='utf-8')
    print(f"\nSaved final results to {final_path}")

    # run analysis
    analyze_results(all_results, output_dir)

# --------------------- Static --------------------
CORPUS_DIR = "./corpus"
TEST_DATASET = "./test_dataset.json"
OUTPUT_DIR = "./results"

CHUNK_RANGES = [
    (200, 300),
    (500, 600),
    (800, 1000)
]

OVERLAP = 0
TOP_K = 5
PERSIST_DIR = None  # set to "./chroma_store" if you want persistence

# -------------------- RUN EVALUATION --------------------

if __name__ == "__main__":
    evaluate_all(
        corpus_dir=CORPUS_DIR,
        test_dataset_path=TEST_DATASET,
        output_dir=OUTPUT_DIR,
        chunk_ranges=CHUNK_RANGES,
        overlap=OVERLAP,
        top_k=TOP_K,
        persist_directory=PERSIST_DIR
    )
# -------------------- CLI --------------------
# if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--corpus_dir', type=str, required=True)
    # parser.add_argument('--test_dataset', type=str, required=True)
    # parser.add_argument('--output_dir', type=str, default='./results')
    # parser.add_argument('--chunk_ranges', type=str, nargs='+', default=['200-300', '500-600', '800-1000'],
    #                     help="Space separated ranges like 200-300 500-600")
    # parser.add_argument('--overlap', type=int, default=0, help="Character overlap between chunks")
    # parser.add_argument('--top_k', type=int, default=5)
    # parser.add_argument('--persist', type=str, default=None, help="Chroma persistence directory (optional)")
    # args = parser.parse_args()

    # # parse chunk ranges
    # parsed = []
    # for r in args.chunk_ranges:
    #     low, high = r.split('-', 1)
    #     parsed.append((int(low), int(high)))

    # evaluate_all(args.corpus_dir, args.test_dataset, args.output_dir, parsed,
    #              overlap=args.overlap, top_k=args.top_k, persist_directory=args.persist)
