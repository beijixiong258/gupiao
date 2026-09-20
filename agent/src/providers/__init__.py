"""LLM provider abstraction layer."""

from typing import Any


def build_llm(*, model_name: str | None = None, callbacks: Any = None) -> Any:
    """仅在真正创建模型客户端时加载模型 SDK。"""
    from src.providers.llm import build_llm as factory

    return factory(model_name=model_name, callbacks=callbacks)

__all__ = ["build_llm"]
