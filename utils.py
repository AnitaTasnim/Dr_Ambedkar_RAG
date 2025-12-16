import os
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

# NLP / embedding / retrieval / llm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# PDF handling
try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None



def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def pdf_to_text(pdf_path: str) -> str:
    """Simple PDF -> text converter using PyPDF2."""
    if PdfReader is None:
        raise RuntimeError("PyPDF2 not installed. Install via `pip install PyPDF2` to parse PDFs.")
    reader = PdfReader(pdf_path)
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n\n".join(pages)


def prepare_corpus(corpus_dir: str) -> Dict[str, str]:
    """
    Load all .txt files from corpus_dir. If PDFs exist, convert them to text
    and save .txt versions.
    Returns dict: filename -> text
    """
    texts = {}
    p = Path(corpus_dir)
    if not p.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    for f in sorted(p.iterdir()):
        if f.suffix.lower() == '.txt':
            texts[f.name] = f.read_text(encoding='utf-8')
        elif f.suffix.lower() == '.pdf':
            txt = pdf_to_text(str(f))
            txt_name = f.stem + '.txt'
            (p / txt_name).write_text(txt, encoding='utf-8')
            texts[txt_name] = txt
    return texts


def smart_chunk_text(text: str, min_chars: int, max_chars: int, overlap: int = 0) -> List[str]:
    """
    Chunk text into windows of size up to max_chars with a minimum of min_chars.
    Attempt to split at sentence boundaries and allow overlap in characters.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current = ""
    for sent in sentences:
        if not sent:
            continue
        if len(current) + len(sent) <= max_chars:
            current = (current + " " + sent).strip()
        else:
            if len(current) >= min_chars:
                chunks.append(current)
                # prepare next chunk with overlap
                if overlap > 0:
                    # take last `overlap` chars of current as prefix for next chunk
                    prefix = current[-overlap:]
                else:
                    prefix = ""
                current = (prefix + " " + sent).strip()
            else:
                # current < min_chars but next sentence pushes over max_chars -> force split
                # attempt to extend until >= min_chars
                current = (current + " " + sent).strip()
                if len(current) >= min_chars:
                    chunks.append(current)
                    current = ""
    if current:
        chunks.append(current)
    # final pass: ensure no chunk > max_chars (split long ones)
    final_chunks = []
    for ch in chunks:
        if len(ch) <= max_chars:
            final_chunks.append(ch)
        else:
            # naive split by chars
            for i in range(0, len(ch), max_chars):
                final_chunks.append(ch[i:i + max_chars])
    return final_chunks


# -------------------- Indexing --------------------
def index_chunks(chunks: List[Dict[str, Any]], embeddings_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 persist_directory: str = None) -> Chroma:
    hf = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    texts = [c['text'] for c in chunks]
    metadatas = [{'source': c['source'], 'chunk_id': c['id']} for c in chunks]
    vectordb = Chroma.from_texts(texts, hf, metadatas=metadatas, persist_directory=persist_directory)
    return vectordb

