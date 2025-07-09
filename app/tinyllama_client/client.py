import asyncio

from fastmcp import Client
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def load_model_sync(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

async def run_chat_agent():
    loop = asyncio.get_running_loop()
    chat_model = await loop.run_in_executor(None, load_model_sync, "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    print("\nSmart Doc Agent ready! Ask questions (type 'exit' to quit).")
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break

        async with Client("http://localhost:8000/mcp/") as client:
            result = await client.call_tool("search", {"query": query, "top_k": 3})
            context_chunks = result.content if hasattr(result, "content") else []
            context = "\n".join(chunk.text for chunk in context_chunks)

        full_prompt = f"Use the following context to answer:\n{context}\n\nQuestion: {query}\nAnswer:"
        response = await loop.run_in_executor(None, lambda: chat_model(full_prompt, max_new_tokens=300))
        print("Bot:", response[0]["generated_text"].split("Answer:")[-1].strip())

if __name__ == "__main__":
    asyncio.run(run_chat_agent())
