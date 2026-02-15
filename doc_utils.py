"""
doc_utils.py

Utility functions for parsing and cleaning PDF, HTML, Markdown, and TXT documents.
"""
from pathlib import Path
from typing import List
import markdown
from bs4 import BeautifulSoup
import pypdf


def parse_txt(file_path: Path) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def parse_md(file_path: Path) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        html = markdown.markdown(f.read())
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator="\n")

def parse_html(file_path: Path) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
        return soup.get_text(separator="\n")

def parse_pdf(file_path: Path) -> str:
    reader = pypdf.PdfReader(str(file_path))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def parse_document(file_path: Path) -> str:
    ext = file_path.suffix.lower()
    if ext == ".txt":
        return parse_txt(file_path)
    elif ext == ".md":
        return parse_md(file_path)
    elif ext == ".html" or ext == ".htm":
        return parse_html(file_path)
    elif ext == ".pdf":
        return parse_pdf(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

def read_documents(corpus_dir: Path) -> List[dict]:
    docs = []
    for file in corpus_dir.glob("*.*"):
        try:
            content = parse_document(file)
            docs.append({
                "content": content,
                "source": file.name
            })
        except Exception as e:
            print(f"Failed to parse {file}: {e}")
    return docs
