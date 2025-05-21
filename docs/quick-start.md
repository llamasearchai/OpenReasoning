# Quick Start Guide

This guide will help you get started with OpenReasoning quickly.

## Installation

Install OpenReasoning using pip:

```bash
pip install openreasoning
```

Or from source:

```bash
git clone https://github.com/OpenReasoning/OpenReasoning.git
cd OpenReasoning
pip install -e .
```

## Setting Up API Keys

OpenReasoning supports multiple model providers. Set up API keys for the providers you want to use:

```bash
# OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# Anthropic
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# HuggingFace
export HUGGINGFACE_API_KEY="your-huggingface-api-key"
```

Alternatively, create a `.env` file in your project directory:

```
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
HUGGINGFACE_API_KEY=your-huggingface-api-key
```

## Basic Usage

### CLI

The fastest way to get started is using the CLI:

```bash
# Check your environment and setup
openreasoning check

# Run the demo to see capabilities
openreasoning demo

# Start a chat session
openreasoning chat

# Start the API server
openreasoning server
```

### Python API

Here's a simple example of using OpenReasoning in Python:

```python
from openreasoning.models.providers import PROVIDERS
from openreasoning.models.base import ModelInput

# Initialize a model provider
provider = PROVIDERS["openai"]()

# Create a model input
input_data = ModelInput(
    prompt="Explain the concept of retrieval-augmented generation in 3 sentences.",
    temperature=0.7,
    max_tokens=150
)

# Generate text
response = provider.generate(input_data)

# Print the response
print(response.text)
```

### Using Retrieval

To use the retrieval components:

```python
import asyncio
from openreasoning.retrieval.base import Document, SearchQuery
from openreasoning.retrieval.providers import get_retriever

async def main():
    # Initialize a retriever
    retriever = get_retriever("chroma")
    
    # Add documents
    documents = [
        Document(content="OpenReasoning is a framework for AI reasoning.", 
                 metadata={"type": "framework", "topic": "AI"}),
        Document(content="Retrieval-augmented generation combines search with text generation.", 
                 metadata={"type": "technique", "topic": "RAG"})
    ]
    
    doc_ids = await retriever.add_documents(documents)
    
    # Search for documents
    query = SearchQuery(text="What is RAG?", k=1)
    results = await retriever.search(query)
    
    print(results[0].content)

asyncio.run(main())
```

### Using Agents

To use agent components:

```python
import asyncio
from openreasoning.agents.base import AgentRequest
from openreasoning.agents.providers import get_agent

async def main():
    # Initialize an agent
    agent = get_agent("reasoning")
    
    # Create a request
    request = AgentRequest(
        prompt="What is the capital of France and what is its population?",
        context="Please provide accurate information with recent statistics."
    )
    
    # Run the agent
    response = await agent.run(request)
    
    print(response.answer)
    
    if response.reasoning:
        print("\nReasoning:")
        print(response.reasoning)
        
    if response.tool_calls:
        print("\nTool Calls:")
        for tool_call in response.tool_calls:
            print(f"- {tool_call}")

asyncio.run(main())
```

## Apple Silicon Optimization

If you're using an Apple M3 chip, you can apply hardware optimizations:

```python
from openreasoning.utils.m3_optimizer import m3_optimizer

# Apply M3 optimizations
m3_optimizer.apply_optimizations()

# Check optimization status
status = m3_optimizer.get_optimization_status()
print(status)
```

## Next Steps

- Check out the [examples](../notebooks/) for more advanced usage
- Read the [API documentation](api/README.md) for detailed reference
- Explore the [M3 optimization features](m3_optimization.md) if you're using Apple Silicon 