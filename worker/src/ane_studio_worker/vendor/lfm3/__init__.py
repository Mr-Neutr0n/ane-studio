"""Pinned local LFM3 implementation. See provenance.json and README.md."""

from __future__ import annotations


def register() -> None:
    """Register reviewed local classes without enabling remote model code."""
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

    from .configuration_lfm3 import Lfm3Config
    from .modeling_lfm3 import Lfm3ForCausalLM, Lfm3Model

    AutoConfig.register("lfm3", Lfm3Config, exist_ok=True)
    AutoModel.register(Lfm3Config, Lfm3Model, exist_ok=True)
    AutoModelForCausalLM.register(Lfm3Config, Lfm3ForCausalLM, exist_ok=True)
