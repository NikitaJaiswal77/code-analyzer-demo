# Python Project Metrics Analyzer

This is a learning-focused static analysis tool built as part of an internship task.

The goal of this project is not perfection, but to understand how
codebases can be analyzed using simple rules and metrics.

---

## 🎯 What does this tool do?

- Scans Python files in a project
- Collects:
  - Number of functions
  - Lines of code
  - Imported modules
- Flags files as "complex" using configurable rules
- Outputs results in JSON format

---

## 🧠 Learning Journey

Day 1:
- Understood what static code analysis means
- Explored how large projects like ERPNext organize code
- Implemented file scanning using Python
- Learned how to count functions and imports


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

