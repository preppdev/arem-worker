"""Shared 4-bit NF4 loader for FLUX pipelines on a 24 GB card.

FLUX.1 is 12B params (~24 GB at bf16) — it does not fit on a 3090
alongside the T5 text encoder + VAE + activations. Sequential CPU
offload fits but is ~5-8× slower because it shuffles every layer between
CPU and GPU each forward pass.

4-bit NF4 quantization of the transformer (~7 GB) + T5 (~5 GB) fits
comfortably with ``enable_model_cpu_offload`` (which keeps the active
component fully resident on GPU), giving a large speedup. This is also
production-representative: prod would run on this same 3090.
"""
from __future__ import annotations

import torch  # type: ignore


def load_flux_nf4(pipeline_cls, model_path: str, transformer_subfolder: str = "transformer"):
    """Build a FLUX pipeline with the transformer + T5 quantized to NF4.

    `pipeline_cls` is the diffusers pipeline class (FluxFillPipeline,
    FluxKontextPipeline, …). Returns the pipeline with model CPU offload
    enabled.
    """
    from diffusers import BitsAndBytesConfig as DiffusersBnB  # type: ignore
    from transformers import BitsAndBytesConfig as TransformersBnB  # type: ignore
    from transformers import T5EncoderModel  # type: ignore

    # Resolve the transformer class from the target pipeline.
    from diffusers import FluxTransformer2DModel  # type: ignore

    t_quant = DiffusersBnB(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        model_path,
        subfolder=transformer_subfolder,
        quantization_config=t_quant,
        torch_dtype=torch.bfloat16,
    )

    te_quant = TransformersBnB(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    text_encoder_2 = T5EncoderModel.from_pretrained(
        model_path,
        subfolder="text_encoder_2",
        quantization_config=te_quant,
        torch_dtype=torch.bfloat16,
    )

    pipe = pipeline_cls.from_pretrained(
        model_path,
        transformer=transformer,
        text_encoder_2=text_encoder_2,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass
    return pipe
