"""LLM interface. Swap providers by setting LLM_PROVIDER=mock|anthropic|openai."""
import json
import os
import re


class LLM:
    def complete(self, system, user):
        raise NotImplementedError


class AnthropicLLM(LLM):
    def __init__(self, model=None):
        import anthropic  # pip install anthropic
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")

    def complete(self, system, user):
        r = self.client.messages.create(
            model=self.model, max_tokens=1500, system=system,
            messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in r.content if b.type == "text")


class OpenAILLM(LLM):
    def __init__(self, model=None):
        from openai import OpenAI  # pip install openai
        self.client = OpenAI()  # reads OPENAI_API_KEY
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def complete(self, system, user):
        r = self.client.chat.completions.create(
            model=self.model, temperature=0, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        return r.choices[0].message.content


def get_llm(provider=None):
    provider = (provider or os.getenv("LLM_PROVIDER", "mock")).lower()
    if provider == "anthropic":
        return AnthropicLLM()
    if provider == "openai":
        return OpenAILLM()
    from mock_llm import MockLLM
    return MockLLM()


def extract_json(text):
    """Parse a JSON object from model output, tolerating code fences and stray prose."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found")
    return json.loads(text[start:end + 1])


def call_json(llm, system, payload, attempts=2):
    user = json.dumps(payload, ensure_ascii=False)
    last = None
    for _ in range(attempts):
        try:
            return extract_json(llm.complete(system, user))
        except Exception as e:  # malformed JSON: ask once more
            last = e
            user += "\n\nYour last reply was not valid JSON. Reply with JSON only."
    raise ValueError(f"LLM did not return valid JSON: {last}")
