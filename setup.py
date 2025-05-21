from setuptools import setup, find_packages

setup(
    name="openreasoning",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.0.267",
        "dspy-ai>=2.0.0",
        "openai>=1.0.0",
        "anthropic>=0.5.0",
        "instructor>=0.3.0",
        "pydantic>=2.0.0",
        "llama-index>=0.8.0",
        "haystack-ai>=2.0.0",
    ],
    python_requires=">=3.9",
    author="OpenReasoning Team",
    author_email="team@openreasoning.ai",
    description="Advanced Multimodal AI Reasoning Framework",
    keywords="ai, llm, reasoning, multimodal, rag",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    entry_points={
        "console_scripts": [
            "openreasoning=openreasoning.cli:app",
        ],
    },
) 