"""
FastAPI server for OpenReasoning.
"""

import os
from typing import Any, Dict, List, Optional, Union

import uvicorn
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
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger
from pydantic import BaseModel, Field

from openreasoning import __version__
from openreasoning.core.config import settings
from openreasoning.models.base import ModelInput, ModelOutput
from openreasoning.models.providers import PROVIDERS
from openreasoning.utils.m3_optimizer import m3_optimizer

# Initialize FastAPI app
app = FastAPI(
    title="OpenReasoning API",
    description="API for the OpenReasoning framework",
    version=__version__,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Apply M3 optimizations if applicable
if settings.is_m3_chip:
    m3_optimizer.apply_optimizations()


# API Models
class CompletionRequest(BaseModel):
    """Request for text completion."""

    prompt: Union[str, List[Dict[str, str]]] = Field(
        ..., description="The prompt or message list"
    )
    provider: str = Field("openai", description="The model provider to use")
    model: Optional[str] = Field(None, description="The specific model to use")
    temperature: float = Field(0.7, description="Temperature for sampling")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    stop_sequences: Optional[List[str]] = Field(
        None, description="Sequences that stop generation"
    )
    model_params: Optional[Dict[str, Any]] = Field(
        None, description="Additional model parameters"
    )


class CompletionResponse(BaseModel):
    """Response from text completion."""

    text: str = Field(..., description="The generated text")
    usage: Dict[str, int] = Field(..., description="Token usage statistics")
    model_info: Dict[str, Any] = Field(
        ..., description="Information about the model used"
    )


class HealthResponse(BaseModel):
    """API health check response."""

    status: str = Field(..., description="API status")
    version: str = Field(..., description="API version")
    providers: Dict[str, bool] = Field(..., description="Available providers")
    optimizations: Dict[str, Any] = Field(..., description="Applied optimizations")


# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check():
    """Check API health and available providers."""
    provider_status = {}

    for provider_name, provider_class in PROVIDERS.items():
        try:
            provider = provider_class()
            provider_status[provider_name] = provider.api_key is not None
        except Exception:
            provider_status[provider_name] = False

    optimization_status = (
        m3_optimizer.get_optimization_status()
        if settings.is_m3_chip
        else {"optimizations_applied": False, "is_m3_chip": False}
    )

    return HealthResponse(
        status="ok",
        version=__version__,
        providers=provider_status,
        optimizations=optimization_status,
    )


# Completion endpoint
@app.post("/complete", response_model=CompletionResponse, tags=["generation"])
async def complete(request: CompletionRequest):
    """Generate text completion from a prompt."""
    if request.provider not in PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"Provider '{request.provider}' not found. Available providers: {list(PROVIDERS.keys())}",
        )

    provider_class = PROVIDERS[request.provider]

    try:
        provider = provider_class(model=request.model)

        if not provider.api_key:
            raise HTTPException(
                status_code=400,
                detail=f"No API key found for provider: {request.provider}",
            )

        input_data = ModelInput(
            prompt=request.prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stop_sequences=request.stop_sequences,
            model_params=request.model_params or {},
        )

        response = provider.generate(input_data)

        return CompletionResponse(
            text=response.text, usage=response.usage, model_info=response.model_info
        )

    except Exception as e:
        logger.error(f"Error during completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Models endpoint
@app.get("/models/{provider}", tags=["models"])
async def list_models(provider: str):
    """List available models for a provider."""
    if provider not in PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"Provider '{provider}' not found. Available providers: {list(PROVIDERS.keys())}",
        )

    provider_class = PROVIDERS[provider]

    try:
        provider_instance = provider_class()
        models = provider_instance.get_available_models()
        return {"provider": provider, "models": models}

    except Exception as e:
        logger.error(f"Error listing models for {provider}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Providers endpoint
@app.get("/providers", tags=["models"])
async def list_providers():
    """List all available providers."""
    return {"providers": list(PROVIDERS.keys())}


# Root endpoint
@app.get("/", tags=["system"])
async def root():
    """API root endpoint."""
    return {
        "name": "OpenReasoning API",
        "version": __version__,
        "description": "API for the OpenReasoning framework",
        "documentation": "/docs",
    }


# Run the API server directly
def run_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """Run the API server."""
    uvicorn.run("openreasoning.api.server:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    # This allows running the server directly with python -m openreasoning.api.server
    run_server()
