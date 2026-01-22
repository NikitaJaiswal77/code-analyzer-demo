# Python Project Metrics Analyzer

This is a learning-focused static analysis tool built as part of an internship task.

The goal of this project is not perfection, but to understand how
codebases can be analyzed using simple rules and metrics.

---
A Python-based Code Analyzer that uses Retrieval-Augmented Generation (RAG) to analyze a Python codebase, explain code behavior in natural language, and detect unimplemented functions.

This project provides an interactive CLI where users can ask questions about their code and receive intelligent, context-aware answers powered by an LLM.

 Features

- Loads and analyzes a Python codebase using FAISS vector storage

- Retrieves relevant code snippets using semantic search

- Uses Flan-T5 to explain code behavior in plain English

- Detects unimplemented functions (pass)

- Explains specific functions like login, logout, helper

- Interactive question–answer CLI interface

- Explanation level configurable via YAM


---


## 📂 Project Structure
project_scanner/
├── scanner.py
├── config.py
├── sample_project/
│ ├── user.py
│ └── utils.py
└── output/
└── report.json

