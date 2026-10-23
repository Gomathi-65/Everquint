"""Optional LLM providers for summarization. Without a key the extractive summarizer is used."""
import os


class AnthropicLLM:
    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")

    def complete(self, system, user):
        r = self.client.messages.create(model=self.model, max_tokens=1200, system=system,
                                        messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in r.content if b.type == "text")


class OpenAILLM:
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def complete(self, system, user):
        r = self.client.chat.completions.create(model=self.model, temperature=0.2, messages=[
            {"role": "system", "content": system}, {"role": "user", "content": user}])
        return r.choices[0].message.content


def get_llm(provider=None):
    """Returns an LLM client, or None for the extractive (no-key) summarizer."""
    provider = (provider or os.getenv("LLM_PROVIDER", "extractive")).lower()
    return {"anthropic": AnthropicLLM, "openai": OpenAILLM}.get(provider, lambda: None)()
