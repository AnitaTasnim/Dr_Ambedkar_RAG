import os
import json
import math
import string
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
 


from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA
from langchain_ollama import OllamaLLM

CORPUS_DIR = "corpus"
TEST_DATASET = "test_dataset.json"


# ------------------------------------------------------------
# Load Test Dataset (FIXED)
# ------------------------------------------------------------
def load_test_dataset():
    with open(TEST_DATASET, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Correct extraction (the FIX)
    if "test_questions" not in data or not isinstance(data["test_questions"], list):
        raise ValueError("❌ test_dataset.json is malformed. Must contain a list under 'test_questions'.")

    return data["test_questions"]


# ------------------------------------------------------------
# Load corpus (same as before)
# ------------------------------------------------------------
def load_documents():
    documents = {}

    for filename in sorted(os.listdir(CORPUS_DIR)):
        if filename.endswith(".txt"):
            path = os.path.join(CORPUS_DIR, filename)
            with open(path, "r", encoding="utf-8") as f:
                documents[filename] = f.read()

    print(f"{len(documents)} documents loaded from corpus/")
    return documents


# ------------------------------------------------------------
# Flexible chunking
# ------------------------------------------------------------
def chunk_documents(docs, chunk_size, overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = []
    metadata = []

    for filename, content in docs.items():
        chunks_split = text_splitter.split_text(content)
        for chunk in chunks_split:
            chunks.append(chunk)
            metadata.append({"source": filename})

    print(f"Chunks created: {len(chunks)}")
    return chunks, metadata


# ------------------------------------------------------------
# Build vector store
# ------------------------------------------------------------
def build_vector_store(chunks, metadata):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_texts(texts=chunks, metadatas=metadata, embedding=embeddings)
    return vectordb


# ------------------------------------------------------------
# RAG Answering
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA

def generate_answer(query, retriever):
    llm = OllamaLLM(model="llama3.2", temperature=0)

    # Create a RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    # Run the chain
    result = qa_chain.invoke({"query": query})

    # result is a dict: {'result': answer_text, 'source_documents': [docs]}
    answer = result["result"]
    retrieved_docs = result["source_documents"]

    return answer, retrieved_docs

 
# ------------------------------------------------------------
 

# ------------------------------------------------------------
# Evaluation Metrics
# ------------------------------------------------------------
def hit_rate(gt_sources, retrieved_docs):
    retrieved = [d.metadata["source"] for d in retrieved_docs]
    return int(any(src in retrieved for src in gt_sources))


def mrr(gt_sources, retrieved_docs):
    retrieved = [d.metadata["source"] for d in retrieved_docs]
    for idx, src in enumerate(retrieved):
        if src in gt_sources:
            return 1 / (idx + 1)
    return 0.0


def precision_at_k(gt_sources, retrieved_docs, k=3):
    retrieved = [d.metadata["source"] for d in retrieved_docs[:k]]
    hits = sum([1 for src in retrieved if src in gt_sources])
    return hits / k


def rouge_l(pred, ref):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(ref, pred)["rougeL"].fmeasure


def bleu_score(pred, ref):
    smoothie = SmoothingFunction().method1
    return sentence_bleu([ref.split()], pred.split(), smoothing_function=smoothie)


def cosine_sim(pred, ref):
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    v1 = embed.embed_query(pred)
    v2 = embed.embed_query(ref)
    return float(cosine_similarity([v1], [v2])[0][0])


# ------------------------------------------------------------
# Main Evaluation per Chunking Strategy
# ------------------------------------------------------------
def evaluate_configuration(chunk_size, overlap, test_questions):
    print(f"\n=== Running evaluation for CHUNK={chunk_size}, OVERLAP={overlap} ===")

    docs = load_documents()
    chunks, metadata = chunk_documents(docs, chunk_size, overlap)
    vectordb = build_vector_store(chunks, metadata)

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    scores = {
        "hit_rate": [],
        "mrr": [],
        "precision@3": [],
        "rougeL": [],
        "bleu": [],
        "cosine": []
    }

    for item in test_questions:
        qid = item["id"]
        q = item["question"]
        gt = item["ground_truth"]
        gt_docs = item["source_documents"]

        pred, retrieved = generate_answer(q, retriever)

        scores["hit_rate"].append(hit_rate(gt_docs, retrieved))
        scores["mrr"].append(mrr(gt_docs, retrieved))
        scores["precision@3"].append(precision_at_k(gt_docs, retrieved))
        scores["rougeL"].append(rouge_l(pred, gt))
        scores["bleu"].append(bleu_score(pred, gt))
        scores["cosine"].append(cosine_sim(pred, gt))

    return {
        "chunk_size": chunk_size,
        "overlap": overlap,
        "hit_rate": float(np.mean(scores["hit_rate"])),
        "mrr": float(np.mean(scores["mrr"])),
        "precision@3": float(np.mean(scores["precision@3"])),
        "rougeL": float(np.mean(scores["rougeL"])),
        "bleu": float(np.mean(scores["bleu"])),
        "cosine": float(np.mean(scores["cosine"]))
    }


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    test_questions = load_test_dataset()

    configurations = [
        (250, 50),
        (550, 50),
        (900, 100)
    ]

    all_results = []

    for chunk, ov in configurations:
        res = evaluate_configuration(chunk, ov, test_questions)
        all_results.append(res)

    with open("test_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)

    print("\n✔ Results written to test_results.json")


if __name__ == "__main__":
    main()
