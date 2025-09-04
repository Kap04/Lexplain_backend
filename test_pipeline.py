import os
from pipeline import extract_text_from_pdf, chunk_text, embed_text, generate_summary

def test_extract_text():
    sample_pdf = "sample.pdf"  # Place a small sample PDF in backend/
    if not os.path.exists(sample_pdf):
        print("No sample.pdf found, skipping.")
        return
    pages = extract_text_from_pdf(sample_pdf)
    assert isinstance(pages, list)
    print("Extracted pages:", pages)

def test_chunk_text():
    pages = [{"page": 1, "text": "This is a test. " * 100}]
    chunks = chunk_text(pages)
    assert len(chunks) > 0
    print("Chunks:", chunks)

def test_embed_text():
    emb = embed_text("test text")
    assert isinstance(emb, list) and len(emb) == 768
    print("Embedding shape:", len(emb))

def test_generate_summary():
    chunks = [{"text": "Test chunk", "startPage": 1, "endPage": 1, "tokens": 10}]
    summary = generate_summary(chunks)
    assert "bullets" in summary and "risks" in summary
    print("Summary:", summary)

if __name__ == "__main__":
    test_extract_text()
    test_chunk_text()
    test_embed_text()
    test_generate_summary()
