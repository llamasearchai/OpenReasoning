"""
API routes for OpenReasoning.
"""

import base64
import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..agents.reasoning import MultiAgentReasoning, ReasoningAgent
from ..core.config import settings
from ..core.logging import configure_logging
from ..models.anthropic import AnthropicModel
from ..models.openai import OpenAIModel
from ..utils.m3_optimizer import m3_optimizer

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)

# Apply M3 optimizations if applicable
if settings.is_m3_chip:
    m3_optimizer.apply_optimizations()

# Create FastAPI app
app = FastAPI(
    title="OpenReasoning API",
    description="API for OpenReasoning - Advanced Multimodal AI Reasoning Framework",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request models
class ImageData(BaseModel):
    """Image data for multimodal requests."""

    url: Optional[str] = None
    base64_data: Optional[str] = None


class CompletionRequest(BaseModel):
    """Request model for completions."""

    messages: List[Dict[str, Any]]
    model: Optional[str] = None
    provider: str = "openai"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stream: bool = False
    images: Optional[List[ImageData]] = None


class ReasoningRequest(BaseModel):
    """Request model for reasoning."""

    query: str
    system_prompt: Optional[str] = None
    context: Optional[str] = None
    history: List[Dict[str, str]] = Field(default_factory=list)
    model: Optional[str] = None
    provider: str = "openai"
    verbose: bool = False
    temperature: float = 0.7
    images: Optional[List[ImageData]] = None
    timeout: Optional[float] = None


class MultiAgentRequest(BaseModel):
    """Request model for multi-agent reasoning."""

    query: str
    context: Optional[str] = None
    history: List[Dict[str, str]] = Field(default_factory=list)
    images: Optional[List[ImageData]] = None
    timeout: Optional[float] = None
    agent_configs: Optional[List[Dict[str, Any]]] = None


# Routes
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "OpenReasoning API",
        "version": "0.1.0",
        "status": "running",
        "m3_optimized": settings.is_m3_chip and m3_optimizer.optimizations_applied,
    }


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """Create a completion."""
    try:
        # Initialize model based on provider
        if request.provider == "openai":
            model = OpenAIModel(model=request.model)
        elif request.provider == "anthropic":
            model = AnthropicModel(model=request.model)
        elif request.provider == "mlx" and settings.is_apple_silicon:
            from ..models.mlx_optimized import MLXModel

            model = MLXModel(
                model_path=request.model or "mlx-community/mistral-7b-instruct-v0.2-q4"
            )
        else:
            raise HTTPException(
                status_code=400, detail=f"Unsupported provider: {request.provider}"
            )

        # Process images if provided
        if request.images:
            # Add images to the last user message
            for i, msg in enumerate(reversed(request.messages)):
                if msg.get("role") == "user":
                    if "images" not in msg:
                        msg["images"] = []
                    for img in request.images:
                        if img.url:
                            msg["images"].append({"url": img.url})
                        elif img.base64_data:
                            msg["images"].append({"base64_data": img.base64_data})
                    break

        # Get completion
        response = model.complete(
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        return response.dict()

    except Exception as e:
        logger.error(f"Error in completion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/reasoning")
async def create_reasoning(request: ReasoningRequest):
    """Create a reasoning chain."""
    try:
        # Process images
        image_urls = []
        if request.images:
            for img in request.images:
                if img.url:
                    image_urls.append(img.url)
                elif img.base64_data:
                    image_urls.append({"base64_data": img.base64_data})

        # Initialize reasoning agent
        agent = ReasoningAgent(
            model_provider=request.provider,
            model_name=request.model,
            verbose=request.verbose,
            temperature=request.temperature,
        )

        # Perform reasoning
        result = agent.reason(
            query=request.query,
            system_prompt=request.system_prompt,
            context=request.context,
            history=request.history,
            image_urls=image_urls if image_urls else None,
            timeout=request.timeout,
        )

        return result

    except Exception as e:
        logger.error(f"Error in reasoning: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/multi-agent")
async def create_multi_agent_reasoning(request: MultiAgentRequest):
    """Create a multi-agent reasoning chain."""
    try:
        # Process images
        image_urls = []
        if request.images:
            for img in request.images:
                if img.url:
                    image_urls.append(img.url)
                elif img.base64_data:
                    image_urls.append({"base64_data": img.base64_data})

        # Use default agent configs if none provided
        agent_configs = request.agent_configs or [
            {
                "id": "researcher",
                "description": "Research specialist with access to search tools",
                "expertise": ["information retrieval", "research", "fact finding"],
                "tool_categories": ["web"],
                "provider": "openai",
                "model": "gpt-4o",
            },
            {
                "id": "analyst",
                "description": "Data analyst with math and calculation expertise",
                "expertise": ["data analysis", "mathematics", "statistics"],
                "tool_categories": ["math"],
                "provider": "openai",
                "model": "gpt-4o",
            },
            {
                "id": "planner",
                "description": "Strategic planner for multi-step reasoning",
                "expertise": ["planning", "step-by-step reasoning", "strategy"],
                "provider": "anthropic" if settings.anthropic_api_key else "openai",
                "model": (
                    "claude-3-opus-20240229" if settings.anthropic_api_key else "gpt-4o"
                ),
            },
        ]

        # Initialize multi-agent reasoning
        multi_agent_system = MultiAgentReasoning(
            agents_config=agent_configs, verbose=True
        )

        # Perform reasoning
        result = multi_agent_system.reason(
            query=request.query,
            context=request.context,
            history=request.history,
            image_urls=image_urls if image_urls else None,
            timeout=request.timeout,
        )

        return result

    except Exception as e:
        logger.error(f"Error in multi-agent reasoning: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """Upload an image and get a base64 representation."""
    try:
        # Read the file
        contents = await file.read()

        # Convert to base64
        base64_encoded = base64.b64encode(contents).decode("utf-8")

        # Generate a temporary ID for the image
        image_id = str(uuid.uuid4())

        return {
            "image_id": image_id,
            "base64_data": base64_encoded,
            "filename": file.filename,
            "content_type": file.content_type,
        }
    except Exception as e:
        logger.error(f"Error uploading image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "m3_optimized": settings.is_m3_chip and m3_optimizer.optimizations_applied,
        "system_info": {
            "python_version": "3.9+",
            "apple_silicon": settings.is_apple_silicon,
            "m3_chip": settings.is_m3_chip,
        },
    }


@app.get("/v1/system-info")
async def system_info():
    """Get system information."""
    if settings.is_m3_chip:
        m3_status = m3_optimizer.get_optimization_status()
    else:
        m3_status = {"available": False}

    try:
        import mlx

        mlx_available = True
        mlx_version = mlx.__version__
    except ImportError:
        mlx_available = False
        mlx_version = None

    return {
        "version": "0.1.0",
        "system": {
            "apple_silicon": settings.is_apple_silicon,
            "m3_chip": settings.is_m3_chip,
            "m3_status": m3_status,
            "environment": settings.environment,
            "mlx_available": mlx_available,
            "mlx_version": mlx_version,
        },
        "api_keys": {
            "openai": bool(settings.openai_api_key),
            "anthropic": bool(settings.anthropic_api_key),
            "huggingface": bool(settings.huggingface_api_key),
        },
    }
