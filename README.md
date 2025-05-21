# OpenReasoning: Advanced Multimodal AI Reasoning Framework

<p align="center">
  <img src="docs/assets/openreasoning.svg" alt="OpenReasoning Logo" width="200"/>
</p>

OpenReasoning is a comprehensive Python framework for developing advanced AI reasoning systems with multimodal capabilities. It integrates state-of-the-art language models, retrieval-augmented generation (RAG), and sophisticated orchestration to enable powerful and flexible AI applications.

## Key Features

- **🧠 Multimodal Reasoning**: Seamlessly combine and reason over text, images, and structured data within your AI pipelines.
- **🔄 Multiple Model Providers**: Easily switch between and utilize models from various providers like OpenAI, Anthropic, Mistral, Hugging Face, and more.
- **🚀 Advanced RAG**: Construct sophisticated RAG systems with optimized prompting, diverse embedding models, and flexible vector store integrations (FAISS, ChromaDB).
- **🤖 Agentic Architecture**: Design and deploy multi-step reasoning agents capable of dynamic tool use and complex problem-solving.
- **🍏 MLX Optimization**: Leverage hardware-accelerated inference on Apple Silicon (M-series chips) for enhanced performance, including specific optimizations for M3 Max.
- **📊 Comprehensive Monitoring & Logging**: Integrated logging with Loguru and Pydantic-based settings for robust tracking of performance, quality, and usage metrics.
- **🔧 Extensible & Modular**: Built with a modular design, allowing for easy customization and extension of core components.
- **命令行界面 (CLI)**: Includes a `colorful-cli` for interactive chat, server management, and demonstrations.

## Quick Start

1.  **Installation:**
    ```bash
    pip install openreasoning
    # Or for development:
    # git clone https://github.com/llamasearchai/OpenReasoning.git
    # cd OpenReasoning
    # pip install -e .
    ```

2.  **Set up API Keys:**
    Create a `.env` file in your project root or export environment variables:
    ```env
    OPENAI_API_KEY="your-openai-api-key"
    ANTHROPIC_API_KEY="your-anthropic-api-key"
    # Add other provider keys as needed
    ```

3.  **Run the Demo:**
    ```bash
    openreasoning demo
    # Or use the interactive chat CLI
    # openreasoning chat
    ```

## Apple Silicon Optimization

OpenReasoning includes special optimizations for Apple Silicon, particularly for M2/M3 series chips:
- Hardware-accelerated tensor operations via MLX integration.
- Optimized memory usage patterns tailored for unified memory architecture.
- Efficient multi-core workload distribution to leverage all performance and efficiency cores.

Check optimization status:
```python
from openreasoning.utils.m3_optimizer import m3_optimizer
print(m3_optimizer.get_optimization_status())
```

## Documentation

For complete documentation, including API references and advanced usage guides, please visit the `docs/` directory (further details TBD as documentation is built out).

## Examples

Explore the `notebooks/` directory for Jupyter notebooks demonstrating key capabilities and use cases, such as:
- Building a multimodal RAG system.
- Creating a custom reasoning agent.
- Fine-tuning models with `dspy-ai` integration.

## Contributing

Contributions are highly welcome! We appreciate any help in improving the framework, adding new features, or enhancing documentation.

Please see `CONTRIBUTING.md` for detailed guidelines on how to contribute.

## License

OpenReasoning is licensed under the [MIT License](LICENSE).

---

*We are actively developing OpenReasoning. Stay tuned for more updates, features, and advanced capabilities!* 