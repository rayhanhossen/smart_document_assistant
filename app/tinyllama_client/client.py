import asyncio
from fastmcp import Client
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def load_hf_chat_model(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

async def run_chat_agent():
    client = Client("http://localhost:8000/mcp/")
    chat_model = load_hf_chat_model()

    # Step 1: Ingest folder (once at the beginning)
    async with client:
        tools = await client.list_tools()
        tool_names = [tool.name for tool in tools]

        if "ingest_folder" in tool_names:
            result = await client.call_tool("ingest_folder", {"path": "/Users/rayhanhossen/Coding Zone/ai_assistant/smart_document_assistant/docs"})
            if hasattr(result, "content"):
                print("Ingest_Folder:", result.content)


    # Step 2: Run Search Chat Loop
    print("\nSmart Doc Agent ready! Ask questions (type 'exit' to quit).")
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break

        async with client:
            result = await client.call_tool("search", {"query": query, "top_k": 3})
            context_chunks = result.content if hasattr(result, "content") else []
            context = "\n".join(chunk.text for chunk in context_chunks)

        prompt = f"Use the following context to answer:\n{context}\n\nQuestion: {query}\nAnswer:"
        response = chat_model(prompt, max_new_tokens=300)[0]["generated_text"]
        print("Bot:", response.split("Answer:")[-1].strip())

if __name__ == "__main__":
    asyncio.run(run_chat_agent())
