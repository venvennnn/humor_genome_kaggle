"""Unified Gemma client with three interchangeable backends.

The goal is that *anyone* — including judges — can run this project regardless of
their setup:

1. ``ollama``  - talks to a local Ollama server running a Gemma model
                 (``ollama run gemma2``). Zero Python ML deps, fastest to try.
2. ``hf``      - loads a Gemma checkpoint via HuggingFace ``transformers``
                 (needs a GPU/accelerator and model access).
3. ``mock``    - a deterministic, offline stand-in that requires no model at
                 all, so the app, CLI, and tests always run. It is clearly
                 labeled as mock output in the UI.

Backend selection is automatic (``auto``): try Ollama, then HF, then fall back
to mock. It can be forced with the ``HUMOR_GENOME_BACKEND`` env var or the
``GemmaConfig.backend`` field.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import requests


class BackendUnavailable(RuntimeError):
    """Raised when a requested backend cannot be initialized."""


DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Gemma family model tags.
#
# Text analysis uses Gemma 3 (the current multimodal generation; the text path
# works identically to Gemma 2's chat format). Set these to `gemma4` / a Gemma 4
# HF id once you have access — nothing else needs to change.
#
# The VIDEO feature needs a model that can ingest images (and ideally audio),
# so it defaults to Gemma 3n, which is natively multimodal (text+image+audio+
# video) and available on Ollama as `gemma3n`. Gemma 3 (4B/12B/27B) also accepts
# images and is a fine substitute for the frame-based path.
DEFAULT_OLLAMA_MODEL = os.environ.get("HUMOR_GENOME_OLLAMA_MODEL", "gemma3")
DEFAULT_HF_MODEL = os.environ.get("HUMOR_GENOME_HF_MODEL", "google/gemma-3-4b-it")
DEFAULT_OLLAMA_VISION_MODEL = os.environ.get(
    "HUMOR_GENOME_OLLAMA_VISION_MODEL", "gemma3n"
)
DEFAULT_HF_VISION_MODEL = os.environ.get(
    "HUMOR_GENOME_HF_VISION_MODEL", "google/gemma-3n-e4b"
)


@dataclass
class GemmaConfig:
    backend: str = field(
        default_factory=lambda: os.environ.get("HUMOR_GENOME_BACKEND", "auto")
    )
    ollama_url: str = DEFAULT_OLLAMA_URL
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    hf_model: str = DEFAULT_HF_MODEL
    ollama_vision_model: str = DEFAULT_OLLAMA_VISION_MODEL
    hf_vision_model: str = DEFAULT_HF_VISION_MODEL
    temperature: float = 0.8
    max_tokens: int = 1024
    request_timeout: int = 180


class GemmaClient:
    """Backend-agnostic text generation over the Gemma model family."""

    def __init__(self, config: Optional[GemmaConfig] = None):
        self.config = config or GemmaConfig()
        self._hf_pipe = None
        self.active_backend: str = "uninitialized"
        self._resolve_backend()

    # ------------------------------------------------------------------ setup
    def _resolve_backend(self) -> None:
        requested = (self.config.backend or "auto").lower()

        if requested == "mock":
            self.active_backend = "mock"
            return

        if requested in ("ollama", "auto"):
            if self._ollama_available():
                self.active_backend = "ollama"
                return
            if requested == "ollama":
                raise BackendUnavailable(
                    f"Ollama not reachable at {self.config.ollama_url}. "
                    "Start it with `ollama run gemma2`."
                )

        if requested in ("hf", "auto"):
            if self._hf_available():
                self.active_backend = "hf"
                return
            if requested == "hf":
                raise BackendUnavailable(
                    "HuggingFace transformers/torch not available or model could "
                    "not be loaded. Install requirements-model.txt."
                )

        # auto fell through: use the always-available mock.
        self.active_backend = "mock"

    def _ollama_available(self) -> bool:
        try:
            resp = requests.get(
                f"{self.config.ollama_url}/api/tags", timeout=3
            )
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def _hf_available(self) -> bool:
        try:
            import torch  # noqa: F401
            from transformers import pipeline  # noqa: F401
        except Exception:
            return False
        try:
            from transformers import pipeline

            self._hf_pipe = pipeline(
                "text-generation",
                model=self.config.hf_model,
                torch_dtype="auto",
                device_map="auto",
            )
            return True
        except Exception:
            self._hf_pipe = None
            return False

    # ------------------------------------------------------------- generation
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        images: Optional[list] = None,
    ) -> str:
        """Return a completion for ``prompt``.

        ``images`` is an optional list of image file paths (JPEG/PNG). They are
        only used by multimodal Gemma backends (Gemma 3 / 3n); the mock ignores
        the pixels but still routes on the prompt. Never raises for mock.
        """
        if self.active_backend == "ollama":
            return self._generate_ollama(prompt, system, images)
        if self.active_backend == "hf":
            return self._generate_hf(prompt, system, images)
        return self._generate_mock(prompt, system)

    @property
    def supports_images(self) -> bool:
        """Whether the active backend can actually see image input."""
        return self.active_backend in ("ollama", "hf")

    @staticmethod
    def _encode_images(images: Optional[list]) -> list:
        import base64

        encoded = []
        for img in images or []:
            try:
                with open(img, "rb") as f:
                    encoded.append(base64.b64encode(f.read()).decode("utf-8"))
            except OSError:
                continue
        return encoded

    def _generate_ollama(
        self, prompt: str, system: Optional[str], images: Optional[list] = None
    ) -> str:
        encoded = self._encode_images(images)
        model = self.config.ollama_vision_model if encoded else self.config.ollama_model
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }
        if system:
            payload["system"] = system
        if encoded:
            payload["images"] = encoded
        resp = requests.post(
            f"{self.config.ollama_url}/api/generate",
            json=payload,
            timeout=self.config.request_timeout,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")

    def _generate_hf(
        self, prompt: str, system: Optional[str], images: Optional[list] = None
    ) -> str:
        content = []
        for img in images or []:
            content.append({"type": "image", "url": img})
        content.append({"type": "text", "text": prompt})
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content if images else prompt})
        out = self._hf_pipe(
            messages,
            max_new_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            do_sample=self.config.temperature > 0,
        )
        generated = out[0]["generated_text"]
        if isinstance(generated, list):
            # chat format returns the full conversation; take the last turn.
            return generated[-1]["content"]
        return str(generated)

    # ------------------------------------------------------------------- mock
    def _generate_mock(self, prompt: str, system: Optional[str]) -> str:
        """Deterministic offline output.

        The mock inspects the prompt to decide whether an analysis or a punch-up
        is being requested and returns schema-valid JSON built from cheap
        heuristics. This keeps the whole pipeline exercisable without a model.
        """
        from .mock_brain import mock_response

        return mock_response(prompt)

    # --------------------------------------------------------------- describe
    def describe(self) -> str:
        if self.active_backend == "ollama":
            return (
                f"Ollama · text={self.config.ollama_model} · "
                f"vision={self.config.ollama_vision_model}"
            )
        if self.active_backend == "hf":
            return (
                f"HuggingFace · text={self.config.hf_model} · "
                f"vision={self.config.hf_vision_model}"
            )
        return "Offline mock (no Gemma weights loaded)"
