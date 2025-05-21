"""
Advanced MLX optimizations for M3 Max.
This module provides enhanced optimizations specific to the M3 Max chip architecture.
"""

import json
import logging
import os
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..core.config import settings
from ..core.models import ModelResponse
from ..utils.m3_optimizer import m3_optimizer

logger = logging.getLogger(__name__)

# Check if MLX is available
try:
    import mlx
    import mlx.core as mx

    MLX_AVAILABLE = True
    logger.info(f"MLX is available (version {mlx.__version__})")
except ImportError:
    MLX_AVAILABLE = False
    logger.warning("MLX is not available. M3-optimized models will not work.")


class M3MaxOptimizedModel:
    """MLX model with M3 Max-specific optimizations."""

    def __init__(
        self,
        model_path: str = "mlx-community/mistral-7b-instruct-v0.2-q4",
        use_metal: bool = True,
        low_memory_mode: bool = False,
    ):
        """Initialize the M3 Max optimized model."""
        if not MLX_AVAILABLE:
            raise ImportError("MLX is not available. Cannot use M3MaxOptimizedModel.")

        if not settings.is_m3_chip:
            logger.warning(
                "This model is optimized for M3 chips but running on a different architecture."
            )

        self.model_path = model_path
        self.use_metal = use_metal
        self.low_memory_mode = low_memory_mode

        # Apply M3 optimizations
        m3_optimizer.apply_optimizations()

        # Set model-specific environment variables
        if use_metal:
            os.environ["MLX_USE_METAL"] = "1"
            # These settings are optimal for M3 Max
            os.environ["MLX_METAL_PREALLOCATE"] = "1"

        if low_memory_mode:
            os.environ["MLX_MEMORY_LIMIT"] = "8GB"  # Conservative limit
        else:
            # Use most of the unified memory but leave some for system
            os.environ["MLX_MEMORY_LIMIT"] = "32GB"  # Good for 64GB M3 Max

        # Lazy loading - model will be loaded on first use
        self._model = None
        self._tokenizer = None

    def _ensure_model_loaded(self):
        """Ensure model is loaded with M3 optimizations."""
        if self._model is None:
            try:
                # Check if mlx_lm is available for loading models
                try:
                    from mlx_lm import generate, load

                    # Set threading optimizations for M3 Max
                    mx.set_num_threads(10)  # Optimal for M3 Max's 10 performance cores

                    logger.info(
                        f"Loading MLX model from {self.model_path} with M3 Max optimizations"
                    )
                    self._model, self._tokenizer = load(self.model_path)

                    # Apply quantization optimizations if available
                    try:
                        from mlx.nn import quantize

                        logger.info("Applying quantization optimizations")
                        self._model = quantize(self._model, quantize.quantize_weights)
                    except (ImportError, AttributeError):
                        logger.info("Quantization optimizations not available")

                    logger.info(
                        "MLX model loaded successfully with M3 Max optimizations"
                    )
                except ImportError:
                    logger.warning("mlx_lm not found, using basic MLX loading")
                    self._load_model_manually()
            except Exception as e:
                logger.error(f"Error loading MLX model: {e}")
                raise

    def _load_model_manually(self):
        """Manually load a model with M3 Max optimizations when mlx_lm is not available."""
        # Implementation would be similar to the basic MLXModel but with M3 Max optimizations
        logger.error("Manual loading with M3 optimizations not implemented yet")
        raise NotImplementedError(
            "Manual loading with M3 optimizations not implemented yet"
        )

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.9,
        batch_size: int = 1,
    ) -> str:
        """Generate text with M3 Max optimizations."""
        try:
            self._ensure_model_loaded()

            start_time = time.time()

            # Generate text
            try:
                from mlx_lm import generate

                # Set better generation parameters for M3 Max
                results = generate(
                    self._model,
                    self._tokenizer,
                    prompt=prompt,
                    temp=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    verbose=False,
                    adapter_name=None,
                )

                end_time = time.time()
                logger.info(
                    f"Text generation completed in {end_time - start_time:.2f} seconds"
                )

                return results
            except ImportError:
                logger.error("mlx_lm not available for generation")
                raise

        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise

    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = 1024,
    ) -> ModelResponse:
        """Get a completion from the model with M3 Max optimizations."""
        try:
            # Convert messages to a prompt
            prompt = self._convert_messages_to_prompt(messages)

            # Generate text
            start_time = time.time()
            generated_text = self.generate(
                prompt=prompt, temperature=temperature, max_tokens=max_tokens or 1024
            )
            end_time = time.time()

            # Create response
            model_response = ModelResponse(
                id=f"mlx-m3-{int(time.time())}",
                model=self.model_path,
                provider="mlx-m3-optimized",
                content=generated_text.strip(),
                usage={"latency_seconds": end_time - start_time},
                metadata={
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "m3_optimized": True,
                    "metal_enabled": self.use_metal,
                    "low_memory_mode": self.low_memory_mode,
                },
            )

            return model_response

        except Exception as e:
            logger.error(f"Error completing M3-optimized request: {e}")
            raise

    def _convert_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to a prompt optimized for the model architecture."""
        # This format works well with Mistral and similar models
        prompt = ""

        for message in messages:
            role = message.get("role", "").lower()
            content = message.get("content", "")

            if role == "system":
                prompt += f"<|system|>\n{content}\n"
            elif role == "user":
                prompt += f"<|user|>\n{content}\n"
            elif role == "assistant":
                prompt += f"<|assistant|>\n{content}\n"
            else:
                # For unknown roles, treat as user
                prompt += f"<|user|>\n{content}\n"

        # Add the final assistant prompt to indicate where the model should generate
        prompt += "<|assistant|>\n"

        return prompt

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the model on M3 Max."""
        stats = {
            "model_path": self.model_path,
            "metal_enabled": self.use_metal,
            "low_memory_mode": self.low_memory_mode,
            "m3_optimizations": m3_optimizer.get_optimization_status(),
            "mlx_version": mlx.__version__ if MLX_AVAILABLE else None,
        }

        # Run a simple benchmark if model is loaded
        if self._model is not None:
            try:
                # Benchmark with a simple prompt
                prompt = "Explain the advantages of the M3 Max chip"

                start_time = time.time()
                _ = self.generate(prompt=prompt, max_tokens=100)
                end_time = time.time()

                stats["benchmark"] = {
                    "prompt_length": len(prompt),
                    "output_tokens": 100,
                    "generation_time": end_time - start_time,
                    "tokens_per_second": 100 / (end_time - start_time),
                }
            except Exception as e:
                stats["benchmark_error"] = str(e)

        return stats
