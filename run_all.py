\
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse

import numpy as np
import pandas as pd
import requests
import torch
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED = 43
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_URL = "https://academic-calendar.wlu.ca/"
START_URL = "https://academic-calendar.wlu.ca/section.php?cal=1&y=93"
MAX_DOCUMENTS = 350
CHUNK_WORDS = 220
CHUNK_OVERLAP = 40
TOP_K = 5
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GENERATOR_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 180
TEMPERATURE = 0.0

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
for d in (RAW_DIR, PROC_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

@dataclass
class Document:
    doc_id: str
    title: str
    url: str
    text: str
    accessed_date: str

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    url: str
    text: str

def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def allowed_calendar_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.netloc != "academic-calendar.wlu.ca":
        return False
    if parsed.path not in {
        "/section.php", "/program.php", "/department.php", "/course.php",
        "/dates.php", "/glossary.php"
    }:
        return False
    query = parsed.query
    return "cal=1" in query and ("y=93" in query or "y=92" in query)

def extract_page(url: str, html: str) -> tuple[str, str, list[str]]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer"]):
        tag.decompose()
    title = normalize_space(soup.title.get_text(" ", strip=True)) if soup.title else url
    main = soup.find("main") or soup.find(id="content") or soup.body or soup
    text = normalize_space(main.get_text(" ", strip=True))
    links = []
    for a in soup.find_all("a", href=True):
        full = urljoin(url, a["href"])
        if allowed_calendar_url(full):
            links.append(full.split("#")[0])
    return title, text, sorted(set(links))

def crawl_corpus() -> list[Document]:
    session = requests.Session()
    session.headers.update({"User-Agent": "WLU-RAG-course-project/1.0 educational use"})
    queue = [START_URL]
    seen = set()
    documents = []
    today = time.strftime("%Y-%m-%d")
    while queue and len(documents) < MAX_DOCUMENTS:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        try:
            response = session.get(url, timeout=25)
            response.raise_for_status()
            title, text, links = extract_page(url, response.text)
            if len(text.split()) >= 60:
                doc_id = "DOC-" + hashlib.sha1(url.encode()).hexdigest()[:10]
                documents.append(Document(doc_id, title, url, text, today))
            for link in links:
                if link not in seen and link not in queue:
                    queue.append(link)
            time.sleep(0.08)
        except Exception as exc:
            print(f"[crawl warning] {url}: {exc}", file=sys.stderr)
    if len(documents) < 200:
        raise RuntimeError(
            f"Only {len(documents)} documents were collected. "
            "Check internet access or increase the crawl starting points."
        )
    with (PROC_DIR / "documents.jsonl").open("w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")
    return documents

def chunk_documents(documents: list[Document]) -> list[Chunk]:
    chunks = []
    step = CHUNK_WORDS - CHUNK_OVERLAP
    for doc in documents:
        words = doc.text.split()
        for start in range(0, len(words), step):
            part = words[start:start + CHUNK_WORDS]
            if len(part) < 35:
                continue
            chunk_id = f"{doc.doc_id}-C{start // step:03d}"
            chunks.append(Chunk(chunk_id, doc.doc_id, doc.title, doc.url, " ".join(part)))
    with (PROC_DIR / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    return chunks

def load_questions() -> list[dict]:
    with (ROOT / "evaluation_questions.json").open(encoding="utf-8") as f:
        return json.load(f)

def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())

class BM25Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.index = BM25Okapi([tokenize(c.text) for c in chunks])

    def search(self, query: str, k: int = TOP_K) -> list[tuple[Chunk, float]]:
        scores = self.index.get_scores(tokenize(query))
        ids = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in ids]

class DenseRetriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.embeddings = self.model.encode(
            [c.text for c in chunks],
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

    def search(self, query: str, k: int = TOP_K) -> list[tuple[Chunk, float]]:
        q = self.model.encode([query], normalize_embeddings=True)
        scores = cosine_similarity(q, self.embeddings)[0]
        ids = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in ids]

class LocalGenerator:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            GENERATOR_MODEL,
            torch_dtype=dtype,
            device_map="auto",
        )
        self.model.eval()

    def answer(self, question: str, retrieved: list[tuple[Chunk, float]] | None) -> str:
        if retrieved:
            context = "\n\n".join(
                f"[{c.chunk_id}] {c.title}\n{c.text}" for c, _ in retrieved
            )
            instruction = (
                "Answer only from the supplied context. Cite every factual claim using "
                "inline chunk citations such as [DOC-abc-C001]. If the context is "
                "insufficient, answer exactly: I don't know."
            )
        else:
            context = "(No retrieved context was supplied.)"
            instruction = "Answer the question concisely from your existing knowledge."
        prompt = (
            f"You are a careful academic-calendar question answering assistant.\n"
            f"{instruction}\n\nCONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:"
        )
        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

def keyword_score(answer: str, required_groups: list[list[str]]) -> float:
    lower = answer.lower()
    if not required_groups:
        return 1.0
    hits = 0
    for group in required_groups:
        if any(term.lower() in lower for term in group):
            hits += 1
    return hits / len(required_groups)

def ground_truth_hit(chunk: Chunk, q: dict) -> bool:
    url_fragments = q.get("source_url_fragments", [])
    keyword_groups = q.get("source_keyword_groups", [])
    url_ok = any(fragment in chunk.url for fragment in url_fragments) if url_fragments else False
    text = chunk.text.lower()
    kw_ok = all(any(term.lower() in text for term in group) for group in keyword_groups) if keyword_groups else False
    return url_ok or kw_ok

def evaluate_retrieval(results: list[tuple[Chunk, float]], q: dict) -> dict:
    if q["type"] == "unanswerable":
        return {"recall_at_5": None, "rr_at_5": None, "first_relevant_rank": None}
    ranks = [i + 1 for i, (chunk, _) in enumerate(results) if ground_truth_hit(chunk, q)]
    return {
        "recall_at_5": 1 if ranks else 0,
        "rr_at_5": (1 / ranks[0]) if ranks else 0,
        "first_relevant_rank": ranks[0] if ranks else None,
    }

def answer_checks(answer: str, q: dict, retrieved: list[tuple[Chunk, float]]) -> dict:
    abstained = answer.strip().lower().startswith("i don't know")
    citation_present = bool(re.search(r"\[DOC-[A-Za-z0-9-]+-C\d+\]", answer))
    cited_ids = re.findall(r"\[(DOC-[A-Za-z0-9-]+-C\d+)\]", answer)
    retrieved_ids = {c.chunk_id for c, _ in retrieved}
    citations_supported = bool(cited_ids) and all(cid in retrieved_ids for cid in cited_ids)
    if q["type"] == "unanswerable":
        correct = abstained
    else:
        correct = keyword_score(answer, q["answer_keyword_groups"]) >= q.get("keyword_threshold", 0.75)
    return {
        "automatic_correct": int(correct),
        "abstained": int(abstained),
        "citation_present": int(citation_present),
        "citations_supported": int(citations_supported),
    }

def run_diagnostic(generator: LocalGenerator, questions: list[dict]) -> None:
    rows = []
    for q in tqdm(questions[:10], desc="No-context diagnostic"):
        answer = generator.answer(q["question"], None)
        score = keyword_score(answer, q.get("answer_keyword_groups", []))
        rows.append({
            "id": q["id"],
            "question": q["question"],
            "answer_without_context": answer,
            "keyword_score": score,
            "likely_correct": int(score >= q.get("keyword_threshold", 0.75)),
        })
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "diagnostic.csv", index=False)

def summarize(df: pd.DataFrame) -> dict:
    summary = {}
    for system in ["bm25", "dense"]:
        part = df[df.system == system]
        answerable = part[part.question_type != "unanswerable"]
        unanswerable = part[part.question_type == "unanswerable"]
        summary[system] = {
            "retrieval_recall_at_5": round(float(answerable.recall_at_5.mean()), 3),
            "retrieval_mrr_at_5": round(float(answerable.rr_at_5.mean()), 3),
            "generation_accuracy": round(float(part.automatic_correct.mean()), 3),
            "citation_presence": round(float(part.citation_present.mean()), 3),
            "citation_support": round(float(part.citations_supported.mean()), 3),
            "unanswerable_abstention_accuracy": round(float(unanswerable.automatic_correct.mean()), 3),
        }
    return summary

def write_results_section(summary: dict, corpus_n: int, chunk_n: int) -> None:
    b = summary["bm25"]
    d = summary["dense"]
    text = f"""# Generated Experimental Results

The crawler collected **{corpus_n} documents**, which produced **{chunk_n} retrievable chunks**.
All systems used top-k = {TOP_K}, {CHUNK_WORDS}-word chunks with {CHUNK_OVERLAP}-word overlap,
the `{EMBEDDING_MODEL}` dense encoder, and the `{GENERATOR_MODEL}` generator.

| System | Recall@5 | MRR@5 | Answer accuracy | Citation support | Unanswerable accuracy |
|---|---:|---:|---:|---:|---:|
| BM25 | {b['retrieval_recall_at_5']:.3f} | {b['retrieval_mrr_at_5']:.3f} | {b['generation_accuracy']:.3f} | {b['citation_support']:.3f} | {b['unanswerable_abstention_accuracy']:.3f} |
| Dense | {d['retrieval_recall_at_5']:.3f} | {d['retrieval_mrr_at_5']:.3f} | {d['generation_accuracy']:.3f} | {d['citation_support']:.3f} | {d['unanswerable_abstention_accuracy']:.3f} |

These automatic results must be manually verified using `per_question_results.csv` before the
final report is submitted. In particular, inspect answer correctness, whether each cited chunk
actually supports the claim, and whether unanswerable questions are handled appropriately.
"""
    (RESULTS_DIR / "generated_results_section.md").write_text(text, encoding="utf-8")

def main() -> None:
    print("1/7 Crawling corpus")
    documents = crawl_corpus()
    print("2/7 Chunking")
    chunks = chunk_documents(documents)
    print(f"Collected {len(documents)} documents and {len(chunks)} chunks.")

    print("3/7 Loading questions and generator")
    questions = load_questions()
    generator = LocalGenerator()

    print("4/7 Running no-context diagnostic")
    run_diagnostic(generator, questions)

    print("5/7 Building retrievers")
    bm25 = BM25Retriever(chunks)
    dense = DenseRetriever(chunks)

    print("6/7 Evaluating systems")
    rows = []
    for system_name, retriever in [("bm25", bm25), ("dense", dense)]:
        for q in tqdm(questions, desc=system_name):
            retrieved = retriever.search(q["question"], TOP_K)
            answer = generator.answer(q["question"], retrieved)
            ret = evaluate_retrieval(retrieved, q)
            checks = answer_checks(answer, q, retrieved)
            rows.append({
                "system": system_name,
                "question_id": q["id"],
                "question_type": q["type"],
                "question": q["question"],
                "reference_answer": q["reference_answer"],
                "generated_answer": answer,
                "retrieved_chunk_ids": " | ".join(c.chunk_id for c, _ in retrieved),
                "retrieved_urls": " | ".join(c.url for c, _ in retrieved),
                **ret,
                **checks,
                "manual_correct": "",
                "manual_citation_supported": "",
                "reviewer_notes": "",
            })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "per_question_results.csv", index=False)
    summary = summarize(df)
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_results_section(summary, len(documents), len(chunks))
    print("7/7 Complete")
    print(json.dumps(summary, indent=2))
    print("Manually review results/per_question_results.csv before reporting final numbers.")

if __name__ == "__main__":
    main()
