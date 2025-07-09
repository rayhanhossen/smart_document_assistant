import asyncio

from fastmcp import Client


async def run_ingest_pdf_client():
    client = Client("http://localhost:8000/mcp/")

    async with client:
        tools = await client.list_tools()
        tool_names = [tool.name for tool in tools]

        if "ingest_folder" in tool_names:
            result = await client.call_tool("ingest_folder", {"path": "/home/rayhanhossen/BracIT/RAG/smart_document_assistant/docs"})
            if hasattr(result, "content"):
                print("Ingest_Folder:", result.content)

if __name__ == "__main__":
    asyncio.run(run_ingest_pdf_client())