"""
MLX-optimized models for Apple Silicon.
"""

import json
import logging
import os
import shutil
import ssl
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..core.config import settings
from ..core.models import ModelResponse
from ..utils.m3_optimizer import m3_optimizer

logger = logging.getLogger(__name__)

# Check if MLX is available (only on Apple Silicon)
try:
    import mlx
    import mlx.core as mx
    from mlx import nn

    MLX_AVAILABLE = True
    logger.info(f"MLX is available (version {mlx.__version__})")
except ImportError:
    MLX_AVAILABLE = False
    logger.warning("MLX is not available. MLX-optimized models will not work.")


class MLXModel:
    """MLX-optimized model for Apple Silicon."""

    def __init__(
        self,
        model_path: str = "mlx-community/mistral-7b-instruct-v0.2-q4",
        tokenizer_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        apply_optimizations: bool = True,
    ):
        """Initialize MLX model."""
        if not MLX_AVAILABLE:
            raise ImportError("MLX is not available. Cannot use MLXModel.")

        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.cache_dir = cache_dir or os.path.expanduser("~/.mlx_models")

        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)

        # Apply M3 optimizations if requested
        if apply_optimizations and settings.is_m3_chip:
            m3_optimizer.apply_optimizations()

        # Lazy loading - model will be loaded on first use
        self._model = None
        self._tokenizer = None
        self._is_multimodal = (
            "llava" in model_path.lower() or "moe" in model_path.lower()
        )

    def _ensure_model_loaded(self):
        """Ensure model is loaded."""
        if self._model is None:
            try:
                # Check if we have mlx_lm for loading models
                try:
                    from mlx_lm import generate, load

                    logger.info(f"Loading MLX model from {self.model_path}")
                    self._model, self._tokenizer = load(
                        self.model_path, cache_dir=self.cache_dir
                    )
                    logger.info("MLX model loaded successfully")
                except ImportError:
                    # Fallback to manual loading
                    logger.warning("mlx_lm not found, attempting manual model loading")
                    self._load_model_manually()
            except Exception as e:
                logger.error(f"Error loading MLX model: {e}")
                raise

    def _load_model_manually(self):
        """Manually load a model when mlx_lm is not available."""
        try:
            import mlx.core as mx
            from transformers import AutoTokenizer

            # Load the tokenizer
            logger.info(f"Loading tokenizer from {self.tokenizer_path}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

            # Download and load the model weights
            model_dir = os.path.join(self.cache_dir, os.path.basename(self.model_path))
            os.makedirs(model_dir, exist_ok=True)

            # Check if it's a local path or a repo ID
            if os.path.exists(self.model_path):
                # Local path, load directly
                logger.info(f"Loading MLX model from local path: {self.model_path}")
                weights_path = os.path.join(self.model_path, "weights.safetensors")
                if not os.path.exists(weights_path):
                    weights_path = os.path.join(self.model_path, "model.safetensors")

                # Load weights
                weights = mx.load(weights_path)

                # Load config
                with open(os.path.join(self.model_path, "config.json"), "r") as f:
                    config = json.load(f)
            else:
                # Remote repo ID, download first
                logger.info(f"Downloading MLX model from {self.model_path}")
                model_info_url = (
                    f"https://huggingface.co/{self.model_path}/resolve/main/config.json"
                )
                weights_url = f"https://huggingface.co/{self.model_path}/resolve/main/weights.safetensors"

                # Create SSL context to bypass certificate issues
                ssl_context = ssl._create_unverified_context()

                # Download config
                config_path = os.path.join(model_dir, "config.json")
                urllib.request.urlretrieve(
                    model_info_url, config_path, context=ssl_context
                )

                # Load config
                with open(config_path, "r") as f:
                    config = json.load(f)

                # Download weights
                weights_path = os.path.join(model_dir, "weights.safetensors")
                urllib.request.urlretrieve(
                    weights_url, weights_path, context=ssl_context
                )

                # Load weights
                weights = mx.load(weights_path)

            # Initialize the model based on the architecture
            model_type = config.get("model_type", "").lower()

            if "mistral" in model_type:
                from mlx.models.mistral import Mistral, MistralConfig

                model_config = MistralConfig.from_dict(config)
                self._model = Mistral(model_config)
            elif "llama" in model_type:
                from mlx.models.llama import Llama, LlamaConfig

                model_config = LlamaConfig.from_dict(config)
                self._model = Llama(model_config)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            # Load the model weights
            self._model.update(weights)
            logger.info(f"MLX model {self.model_path} loaded successfully")

        except Exception as e:
            logger.error(f"Manual model loading failed: {e}")
            raise

    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = 1024,
    ) -> ModelResponse:
        """Get a completion from the model."""
        try:
            self._ensure_model_loaded()

            # Convert messages to a prompt format the model can understand
            prompt = self._convert_messages_to_prompt(messages)

            start_time = time.time()

            # Generate text
            try:
                from mlx_lm import generate

                # Generate
                generated_text = generate(
                    self._model,
                    self._tokenizer,
                    prompt=prompt,
                    temp=temperature,
                    max_tokens=max_tokens or 1024,
                )
            except ImportError:
                # Fallback to manual generation
                logger.warning("mlx_lm not found, using manual text generation")
                generated_text = self._generate_text_manually(
                    prompt, temperature, max_tokens
                )

            end_time = time.time()

            # Create response
            model_response = ModelResponse(
                id=f"mlx-{int(time.time())}",
                model=self.model_path,
                provider="mlx-local",
                content=generated_text.strip(),
                usage={"latency_seconds": end_time - start_time},
                metadata={
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "optimized_for_m3": settings.is_m3_chip
                    and m3_optimizer.optimizations_applied,
                },
            )

            return model_response

        except Exception as e:
            logger.error(f"Error completing MLX request: {e}")
            raise

    def _generate_text_manually(
        self, prompt: str, temperature: float, max_tokens: int
    ) -> str:
        """Manually generate text when mlx_lm is not available."""
        try:
            import mlx.core as mx

            # Tokenize the prompt
            tokens = mx.array(self._tokenizer.encode(prompt))

            # Generate with temperature
            results = []
            for _ in range(max_tokens):
                logits = self._model(tokens)
                next_token = mx.random.categorical(logits[-1:] / max(0.1, temperature))
                tokens = mx.concatenate([tokens, next_token])

                # Check if we hit the end token
                if next_token.item() in self._tokenizer.eos_token_id:
                    break

                # Add to results
                results.append(next_token.item())

                # Check if reached max tokens
                if len(results) >= max_tokens:
                    break

            # Decode the tokens
            return self._tokenizer.decode(results)
        except Exception as e:
            logger.error(f"Error in manual text generation: {e}")
            raise

    def _convert_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to a prompt for MLX model."""
        prompt = ""

        for message in messages:
            if message["role"] == "system":
                # Add system message at the beginning
                prompt += f"<|system|>\n{message['content']}\n"
            elif message["role"] == "user":
                prompt += f"<|user|>\n{message['content']}\n"
            elif message["role"] == "assistant":
                prompt += f"<|assistant|>\n{message['content']}\n"

        # Add the final assistant prompt to indicate where the model should generate
        prompt += "<|assistant|>\n"

        return prompt


class MLXMultiModalModel(MLXModel):
    """MLX-optimized multimodal model for Apple Silicon."""

    def __init__(
        self,
        model_path: str = "mlx-community/llava-1.5-7b-mlx",
        tokenizer_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        apply_optimizations: bool = True,
    ):
        """Initialize MLX multimodal model."""
        super().__init__(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            cache_dir=cache_dir,
            apply_optimizations=apply_optimizations,
        )
        self._image_processor = None

    def _ensure_model_loaded(self):
        """Ensure model and image processor are loaded."""
        if self._model is None:
            try:
                # Load image processor
                try:
                    from transformers import CLIPImageProcessor

                    self._image_processor = CLIPImageProcessor.from_pretrained(
                        self.model_path
                    )
                except:
                    logger.warning(
                        "Could not load CLIPImageProcessor, image processing may not work correctly"
                    )

                # Load the model using the parent method
                super()._ensure_model_loaded()

            except Exception as e:
                logger.error(f"Error loading MLX multimodal model: {e}")
                raise

    def _process_images(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process images in messages and return processed images and updated messages."""
        if not self._image_processor:
            logger.warning("Image processor not available, images will be ignored")
            return {}, messages

        import base64
        import io

        import numpy as np
        from PIL import Image as PILImage

        processed_images = {}
        updated_messages = []

        for msg in messages:
            if msg["role"] == "user" and "images" in msg:
                images = msg.pop("images", [])
                image_placeholders = []

                for i, img in enumerate(images):
                    image_id = f"img_{int(time.time())}_{i}"

                    try:
                        # Load image from various sources
                        if isinstance(img, dict):
                            if "url" in img:
                                with urllib.request.urlopen(img["url"]) as response:
                                    image_data = response.read()
                                    pil_image = PILImage.open(io.BytesIO(image_data))
                            elif "file_path" in img:
                                pil_image = PILImage.open(img["file_path"])
                            elif "base64_data" in img:
                                image_data = base64.b64decode(img["base64_data"])
                                pil_image = PILImage.open(io.BytesIO(image_data))
                            else:
                                logger.warning(f"Unsupported image format: {img}")
                                continue
                        elif isinstance(img, str):
                            # Assume it's a URL
                            with urllib.request.urlopen(img) as response:
                                image_data = response.read()
                                pil_image = PILImage.open(io.BytesIO(image_data))
                        else:
                            logger.warning(f"Unsupported image type: {type(img)}")
                            continue

                        # Process the image
                        processed_image = self._image_processor(
                            pil_image, return_tensors="np"
                        )
                        processed_images[image_id] = processed_image

                        # Add placeholder to message
                        image_placeholders.append(f"<image:{image_id}>")

                    except Exception as e:
                        logger.error(f"Error processing image: {e}")

                # Update message with image placeholders
                if image_placeholders:
                    content = msg.get("content", "")
                    msg["content"] = content + "\n" + "\n".join(image_placeholders)

                updated_messages.append(msg)
            else:
                updated_messages.append(msg)

        return processed_images, updated_messages

    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = 1024,
    ) -> ModelResponse:
        """Get a completion from the model with image support."""
        try:
            self._ensure_model_loaded()

            # Process images in messages
            processed_images, updated_messages = self._process_images(messages)

            # Convert messages to a prompt format the model can understand
            prompt = self._convert_messages_to_prompt(updated_messages)

            # Replace image placeholders with actual image features
            # This would depend on the specific multimodal MLX model implementation
            # For now, this is a placeholder implementation

            start_time = time.time()

            # Generate text
            try:
                # Custom multimodal generation logic would go here
                # For now, fall back to text-only generation
                generated_text = (
                    "Multimodal generation is not fully implemented yet in MLX."
                )
            except Exception as e:
                logger.error(f"Error in multimodal generation: {e}")
                generated_text = "Error generating multimodal response."

            end_time = time.time()

            # Create response
            model_response = ModelResponse(
                id=f"mlx-mm-{int(time.time())}",
                model=self.model_path,
                provider="mlx-multimodal",
                content=generated_text.strip(),
                usage={"latency_seconds": end_time - start_time},
                metadata={
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "image_count": len(processed_images),
                    "optimized_for_m3": settings.is_m3_chip
                    and m3_optimizer.optimizations_applied,
                },
            )

            return model_response

        except Exception as e:
            logger.error(f"Error completing MLX multimodal request: {e}")
            raise
