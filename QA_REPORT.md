# RAG System QA Report

## Project Overview
This project uses a RAG system to answer questions about a Python codebase.

## QA Results

| Question | Expected | Actual Answer | Accuracy | Hallucination |
|--------|----------|--------------|----------|---------------|
| What does login do? | Not implemented | Correct | High | No |
| What does project do? | Code analysis | Correct | High | No |
| Explain ingest.py | Explain flow | Repeated text | Low | No |
| Embeddings? | HuggingFace | "Using python" | Low | No |
| Vector DB? | FAISS | Random code | Low | Yes |
| Hallucination? | Out of context | "in a room" | Low | Yes |

## Conclusion
- Direct code questions work well
- Conceptual questions cause hallucination
- Needs better chunking and prompts
