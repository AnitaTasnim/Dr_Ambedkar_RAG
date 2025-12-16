from typing import List
from rouge_score import rouge_scorer 
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
# smoothing for BLEU
bleu_smooth = SmoothingFunction().method1
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)


# -------------------- Metrics --------------------
def retrieval_hit_rate(retrieved_sources: List[str], gt_sources: List[str]) -> int:
    return 1 if any(s in gt_sources for s in retrieved_sources) else 0


def retrieval_mrr(retrieved_sources: List[str], gt_sources: List[str]) -> float:
    for idx, s in enumerate(retrieved_sources, start=1):
        if s in gt_sources:
            return 1.0 / idx
    return 0.0


def precision_at_k(retrieved_sources: List[str], gt_sources: List[str], k: int) -> float:
    topk = retrieved_sources[:k]
    hits = sum(1 for s in topk if s in gt_sources)
    return hits / k


def compute_rouge_l(pred: str, ref: str) -> float:
    if not pred or not ref:
        return 0.0
    r = scorer.score(ref, pred)
    return r['rougeL'].fmeasure


def compute_bleu(pred: str, ref: str) -> float:
    if not pred or not ref:
        return 0.0
    ref_toks = [word_tokenize(ref)]
    pred_toks = word_tokenize(pred)
    try:
        return sentence_bleu(ref_toks, pred_toks, smoothing_function=bleu_smooth)
    except Exception:
        return 0.0


def compute_cosine_sim(st_model: SentenceTransformer, pred: str, ref: str) -> float:
    if not pred or not ref:
        return 0.0
    emb = st_model.encode([pred, ref])
    return float(cosine_similarity([emb[0]], [emb[1]])[0][0])


def compute_relevance(pred: str, ref: str) -> float:
    if not ref:
        return 0.0
    p = set(word_tokenize(pred.lower()))
    r = set(word_tokenize(ref.lower()))
    if not r:
        return 0.0
    return len(p & r) / len(r)


def compute_faithfulness(pred: str, retrieved_context: str) -> float:
    if not pred or not retrieved_context:
        return 0.0
    p = set(word_tokenize(pred.lower()))
    s = set(word_tokenize(retrieved_context.lower()))
    if not p:
        return 0.0
    return len(p & s) / len(p)

