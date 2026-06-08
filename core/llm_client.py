"""
core/llm_client.py
统一的 LLM 调用客户端，支持 OpenAI、阿里云百练、Anthropic 和 Ollama。
"""

import json
import logging
from typing import Generator, Iterator, Optional

from config.settings import Settings

logger = logging.getLogger(__name__)


class LLMClient:
    """对多 provider 的 LLM 调用做统一封装。"""

    def __init__(self, settings: Settings):
        self.settings = settings
        logger.info(
            "LLMClient 初始化 | provider=%s | model=%s",
            settings.llm_provider,
            settings.model_name,
        )

    def chat(
        self,
        messages: list[dict],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        provider = self.settings.llm_provider
        if provider == "openai":
            return self._call_openai(messages, system_prompt, temperature, max_tokens)
        if provider == "bailian":
            return self._call_bailian(messages, system_prompt, temperature, max_tokens)
        if provider == "anthropic":
            return self._call_anthropic(messages, system_prompt, temperature, max_tokens)
        if provider == "ollama":
            return self._call_ollama(messages, system_prompt, temperature, max_tokens)
        raise ValueError(f"不支持的 provider: {provider}")

    def chat_stream(
        self,
        messages: list[dict],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Generator[str, None, None]:
        provider = self.settings.llm_provider
        if provider == "openai":
            yield from self._stream_openai(messages, system_prompt, temperature, max_tokens)
            return
        if provider == "bailian":
            yield from self._stream_bailian(messages, system_prompt, temperature, max_tokens)
            return
        if provider == "anthropic":
            yield from self._stream_anthropic(messages, system_prompt, temperature, max_tokens)
            return
        if provider == "ollama":
            yield from self._stream_ollama(messages, system_prompt, temperature, max_tokens)
            return
        raise ValueError(f"不支持的 provider: {provider}")

    def _call_openai(self, messages, system_prompt, temperature, max_tokens) -> str:
        return self._call_openai_compatible(
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url,
            model=self.settings.openai_model,
            messages=messages,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _call_bailian(self, messages, system_prompt, temperature, max_tokens) -> str:
        return self._call_openai_compatible(
            api_key=self.settings.bailian_api_key,
            base_url=self.settings.bailian_base_url,
            model=self.settings.bailian_model,
            messages=messages,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _call_openai_compatible(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        messages: list[dict],
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("请安装 openai：pip install openai") from exc

        client = OpenAI(api_key=api_key, base_url=base_url)
        resp = client.chat.completions.create(
            model=model,
            messages=self._prepend_system(messages, system_prompt),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    def _call_anthropic(self, messages, system_prompt, temperature, max_tokens) -> str:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("请安装 anthropic：pip install anthropic") from exc

        client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
        kwargs = {
            "model": self.settings.anthropic_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        resp = client.messages.create(**kwargs)
        return resp.content[0].text

    def _call_ollama(self, messages, system_prompt, temperature, max_tokens) -> str:
        try:
            import httpx
        except ImportError as exc:
            raise ImportError("请安装 httpx：pip install httpx") from exc

        resp = httpx.post(
            f"{self.settings.ollama_base_url}/api/chat",
            json={
                "model": self.settings.ollama_model,
                "messages": self._prepend_system(messages, system_prompt),
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def _stream_openai(self, messages, system_prompt, temperature, max_tokens) -> Iterator[str]:
        yield from self._stream_openai_compatible(
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url,
            model=self.settings.openai_model,
            label="OpenAI",
            messages=messages,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _stream_bailian(self, messages, system_prompt, temperature, max_tokens) -> Iterator[str]:
        yield from self._stream_openai_compatible(
            api_key=self.settings.bailian_api_key,
            base_url=self.settings.bailian_base_url,
            model=self.settings.bailian_model,
            label="Bailian",
            messages=messages,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _stream_openai_compatible(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        label: str,
        messages: list[dict],
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> Iterator[str]:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("请安装 openai：pip install openai") from exc

        client = OpenAI(api_key=api_key, base_url=base_url)
        logger.debug("[%s stream] model=%s", label, model)

        with client.chat.completions.create(
            model=model,
            messages=self._prepend_system(messages, system_prompt),
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        ) as stream:
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

    def _stream_anthropic(self, messages, system_prompt, temperature, max_tokens) -> Iterator[str]:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("请安装 anthropic：pip install anthropic") from exc

        client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
        kwargs = {
            "model": self.settings.anthropic_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        logger.debug("[Anthropic stream] model=%s", self.settings.anthropic_model)
        with client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield text

    def _stream_ollama(self, messages, system_prompt, temperature, max_tokens) -> Iterator[str]:
        try:
            import httpx
        except ImportError as exc:
            raise ImportError("请安装 httpx：pip install httpx") from exc

        url = f"{self.settings.ollama_base_url}/api/chat"
        logger.debug("[Ollama stream] url=%s model=%s", url, self.settings.ollama_model)

        with httpx.stream(
            "POST",
            url,
            json={
                "model": self.settings.ollama_model,
                "messages": self._prepend_system(messages, system_prompt),
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
            timeout=120.0,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                token = data.get("message", {}).get("content", "")
                if token:
                    yield token
                if data.get("done"):
                    break

    @staticmethod
    def _prepend_system(messages: list[dict], system_prompt: Optional[str]) -> list[dict]:
        if not system_prompt:
            return messages
        return [{"role": "system", "content": system_prompt}] + messages
