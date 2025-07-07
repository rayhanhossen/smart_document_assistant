from pathlib import Path
import faiss
import fitz
import numpy as np
from sentence_transformers import SentenceTransformer


EMBED_DIM = 384
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

class PDFRagIngestor:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index, self.doc_map = self._init_index()

    @staticmethod
    def _init_index():
        idx = faiss.IndexFlatL2(EMBED_DIM)
        doc_map = {}
        return idx, doc_map

    @staticmethod
    def _load_text(pdf_file: Path) -> str:
        doc = fitz.open(str(pdf_file))
        return "".join(page.get_text() for page in doc)

    @staticmethod
    def _chunk_text(text: str) -> list[str]:
        chunks = []
        step = CHUNK_SIZE - CHUNK_OVERLAP
        for i in range(0, len(text), step):
            chunks.append(text[i : i + CHUNK_SIZE])
        return chunks

    def _embed_and_add(self, chunks: list[str]):
        vecs = self.model.encode(chunks, convert_to_numpy=True)
        for chunk, vec in zip(chunks, vecs):
            idx = self.index.ntotal
            self.index.add(np.array([vec], dtype='float32'))
            self.doc_map[idx] = chunk

    def ingest_folder(self, folder_path: str) -> str:
        folder = Path(folder_path)
        pdfs = list(folder.glob("*.pdf"))
        for pdf in pdfs:
            text = self._load_text(pdf)
            chunks = self._chunk_text(text)
            self._embed_and_add(chunks)

        return f"Ingested {self.index.ntotal} chunks from {len(pdfs)} PDFs."

    def search(self, query: str, top_k: int = 3) -> list[str]:
        q_emb = self.model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(np.array(q_emb, dtype='float32'), top_k)
        return [self.doc_map[int(i)] for i in I[0]]
