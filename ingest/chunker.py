from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_DIR = Path(__file__).parent.parent / "docs"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50


def load_markdown(filepath: Path) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def load_pdf(filepath: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(filepath))
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def load_doc(filepath: Path) -> str:
    ext = filepath.suffix.lower()
    if ext == ".md":
        return load_markdown(filepath)
    elif ext == ".pdf":
        return load_pdf(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def chunk_docs() -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    all_chunks = []

    for filepath in DOCS_DIR.iterdir():
        if filepath.suffix.lower() not in [".md", ".pdf"]:
            continue

        print(f"Loading: {filepath.name}")
        text = load_doc(filepath)
        chunks = splitter.split_text(text)

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "id": f"{filepath.stem}_{i}",
                "text": chunk,
                "metadata": {
                    "source": filepath.name,
                    "chunk_index": i,
                }
            })

    print(f"Total chunks created: {len(all_chunks)}")
    return all_chunks