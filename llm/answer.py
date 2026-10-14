import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-3-5-sonnet-20241022"

SYSTEM_PROMPT = """You are a Commerce developer assistant.
Answer ONLY using the context provided below.
Always cite the source document name for each piece of information you use.
If the answer is not in the context, say: "I don't have that information in the available documentation."
Be concise and technical."""


def build_context(chunks: list[dict]) -> str:
    parts = []
    for chunk in chunks:
        source = chunk["metadata"]["source"]
        text = chunk["text"]
        parts.append(f"[Source: {source}]\n{text}")
    return "\n\n---\n\n".join(parts)


def generate_answer(query: str, chunks: list[dict]) -> dict:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    context = build_context(chunks)

    user_message = f"""Context:
{context}

Question: {query}"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    sources = list(set(c["metadata"]["source"] for c in chunks))

    return {
        "answer": response.content[0].text,
        "sources": sources,
        "chunks_used": len(chunks),
    }