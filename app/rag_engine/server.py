from fastmcp import FastMCP
from app.rag_engine.ingestor import PDFRagIngestor

mcp = FastMCP("Smart Document Assistant")
ingestor = PDFRagIngestor()

@mcp.tool()
def ingest_folder(path: str) -> str:
    return ingestor.ingest_folder(path)

@mcp.tool()
def search(query: str, top_k: int = 3) -> list[str]:
    return ingestor.search(query, top_k)

if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="localhost", port=8000)