import os
import json
from config import MAX_FUNCTIONS, MAX_LINES

def scan_file(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    imports = []
    functions = 0

    for line in lines:
        line = line.strip()
        if line.startswith("import") or line.startswith("from"):
            imports.append(line)
        if line.startswith("def "):
            functions += 1

    return {
        "lines_of_code": len(lines),
        "functions": functions,
        "imports": imports,
        "is_complex": functions > MAX_FUNCTIONS or len(lines) > MAX_LINES
    }

def scan_project(folder):
    report = {}

    for file in os.listdir(folder):
        if file.endswith(".py"):
            path = os.path.join(folder, file)
            report[file] = scan_file(path)

    return report

result = scan_project("sample_project")

with open("output/report.json", "w") as f:
    json.dump(result, f, indent=4)

print("Project scan completed.")
