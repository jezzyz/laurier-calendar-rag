# Laurier Academic Calendar RAG

**Author:** Jeremy Joanes  
**Student ID:** 169037356

This project builds and evaluates a Retrieval-Augmented Generation (RAG) system over the
Wilfrid Laurier University 2026/2027 Undergraduate Academic Calendar. It compares:

1. Classical sparse retrieval using BM25
2. Dense retrieval using `sentence-transformers/all-MiniLM-L6-v2`
3. Answer generation using the same local instruction-tuned LLM for both retrievers

The system requires the generator to answer only from retrieved context, include inline
chunk citations, and return `I don't know` when context is insufficient.

## Author

This implementation, evaluation set, experimental workflow, and report were prepared by Jeremy Joanes (Student ID 169037356).

## Tested environment

- macOS 13+ or Linux
- Python 3.10 or 3.11
- Approximately 5 GB free disk space
- Internet access on first run to download the corpus and open-source models

## Quick start

```bash
git clone YOUR_REPOSITORY_URL
cd laurier-rag-project
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python run_all.py
```

On Apple Silicon, PyTorch automatically uses MPS when available. The default generator is
`Qwen/Qwen2.5-0.5B-Instruct`, which is small enough for a modern Mac. To use Ollama instead,
see the configuration notes below.

## One-command reproduction

```bash
python run_all.py
```

This command:

1. Crawls 200-500 pages from the Laurier undergraduate calendar
2. Cleans and chunks the documents
3. Runs the no-context diagnostic
4. Builds BM25 and dense indexes
5. Evaluates retrieval and generation on the gold questions
6. Writes `results/per_question_results.csv`
7. Writes `results/summary.json`
8. Writes `results/generated_results_section.md`

## Configuration

Edit constants near the top of `run_all.py`:

- `MAX_DOCUMENTS`
- `TOP_K`
- `CHUNK_WORDS`
- `CHUNK_OVERLAP`
- `GENERATOR_MODEL`
- `EMBEDDING_MODEL`

The same generation model and settings are used for both retrieval systems.

## Output files

- `data/processed/documents.jsonl`
- `data/processed/chunks.jsonl`
- `results/diagnostic.csv`
- `results/per_question_results.csv`
- `results/summary.json`
- `results/generated_results_section.md`

## Evaluation metrics

Retrieval:
- Recall@5
- Mean Reciprocal Rank at 5
- Success@5 for answerable questions

Generation:
- Human-style rule-based correctness check using required answer keywords
- Citation presence
- Citation support
- Correct abstention on unanswerable questions

The per-question CSV should be manually reviewed before submission. The report discloses
that automatic checks were followed by human verification.

## Ethical and reproducibility statement

The evaluation questions were drafted by the student after reviewing the selected calendar
pages. An LLM was used to assist with code and wording; all questions, reference answers,
source mappings, and final judgments must be manually checked by the student before submission.
No evaluation score should be reported unless it was produced by the included code.

## One-click Mac finalization

Double-click `RUN_AND_FINALIZE.command`, or run:

```bash
./RUN_AND_FINALIZE.command
```

After experiments finish, the script creates:

- `submission/Jeremy_Joanes_RAG_Report.docx`
- `submission/Jeremy_Joanes_RAG_Report.pdf` when LibreOffice is installed

Add the final GitHub and video URLs before submitting.
