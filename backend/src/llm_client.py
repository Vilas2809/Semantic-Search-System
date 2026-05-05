from groq import Groq
from config import config


class LLMClient:
    """Groq-backed LLM for answer generation."""

    def __init__(self):
        config.validate()
        self.client = Groq(api_key=config.GROQ_API_KEY)
        self.model = config.GROQ_MODEL

    def generate_answer(self, query: str, context_chunks: list[str], system_prompt: str = None) -> str:
        """Generate an answer grounded in the retrieved context."""
        context = "\n\n---\n\n".join(
            f"[Source {i + 1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
        )

        system = system_prompt or (
            "You are a precise question-answering assistant. "
            "Answer the user's question using ONLY the provided context. "
            "If the answer cannot be found in the context, say so clearly. "
            "Cite source numbers like [Source 1] when referencing specific content."
        )

        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            },
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            max_tokens=1024,
        )
        return response.choices[0].message.content

    def chat(self, message: str, system_prompt: str = None) -> str:
        """Send a message directly to Groq without any document context."""
        system = system_prompt or "You are a helpful assistant. Answer the user's question clearly and concisely."
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": message},
            ],
            temperature=0.7,
            max_tokens=1024,
        )
        return response.choices[0].message.content

    def rerank_with_llm(self, query: str, chunks: list[str], top_k: int = 3) -> list[int]:
        """Ask the LLM to pick the most relevant chunk indices (0-based)."""
        numbered = "\n\n".join(f"[{i}] {chunk}" for i, chunk in enumerate(chunks))
        prompt = (
            f"Query: {query}\n\n"
            f"Passages:\n{numbered}\n\n"
            f"Return a JSON array of the {top_k} most relevant passage indices (0-based), "
            f"ordered by relevance. Example: [2, 0, 4]"
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=64,
        )
        import json, re
        text = response.choices[0].message.content
        match = re.search(r"\[[\d,\s]+\]", text)
        if match:
            indices = json.loads(match.group())
            return [i for i in indices if 0 <= i < len(chunks)][:top_k]
        return list(range(min(top_k, len(chunks))))
