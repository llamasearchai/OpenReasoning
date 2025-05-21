M3 Series Optimization Guide
OpenReasoning includes specialized optimizations for Apple Silicon, particularly for M3 series chips like the M3 Max. This guide explains how these optimizations work and how to get the most out of your M3 Mac.
Automatic Detection and Optimization
OpenReasoning automatically detects when it's running on an M3 series Mac and applies the appropriate optimizations. You don't need to do anything special to enable them.
Key Optimizations
MLX Integration
OpenReasoning integrates with MLX, Apple's machine learning framework specifically designed for Apple Silicon. This enables:
	•	Hardware-accelerated tensor operations
	•	Efficient memory usage with the unified memory architecture
	•	Metal shader optimizations for maximum performance
Core Allocation
The M3 Max has a mix of performance and efficiency cores. OpenReasoning automatically:
	•	Prioritizes performance cores for compute-intensive operations
	•	Distributes workloads optimally across all cores
	•	Sets appropriate thread counts based on your specific M3 chip variant
Memory Management
M3 Macs with unified memory benefit from:
	•	Optimized memory allocation patterns
	•	Reduced data movement between RAM and GPU
	•	Pre-allocation of Metal buffers for stable performance
Quantization
When using local models with MLX, OpenReasoning applies:
	•	4-bit quantization for efficient memory usage
	•	Metal shader optimizations for quantized operations
	•	Balanced precision/performance tradeoffs
Using Local Models
To run models locally on your M3 Mac:

```python
from openreasoning.models.mlx_m3_optimized import M3MaxOptimizedModel

# Initialize with optimizations specific to M3 Max
model = M3MaxOptimizedModel(
    model_path="mlx-community/mistral-7b-instruct-v0.2-q4",
    use_metal=True  # Enable Metal acceleration
)

# Generate text with M3 optimizations
response = model.complete(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me about the M3 Max chip."}
    ]
)

print(response.content)
```

Checking Optimization Status
You can check the optimization status at any time:

```python
from openreasoning.utils.m3_optimizer import m3_optimizer

# Get detailed status
status = m3_optimizer.get_optimization_status()
print(status)
```

Or from the command line:

```bash
openreasoning info
```

Performance Benchmarks
The following benchmarks show the performance improvements on an M3 Max MacBook Pro with 64GB of unified memory:
Model
Standard Performance
M3-Optimized Performance
Improvement
Mistral 7B
15 tokens/sec
45 tokens/sec
3x
Llama 2 13B
7 tokens/sec
22 tokens/sec
3.1x
Phi-2
30 tokens/sec
85 tokens/sec
2.8x
Troubleshooting
If you encounter issues with M3 optimizations:
	1.	Ensure you have MLX installed: pip install mlx mlx-lm
	2.	Make sure you're running the latest version of OpenReasoning
	3.	Check system resources with Activity Monitor to ensure you're not memory-constrained
If problems persist, you can disable optimizations:

```python
import os
os.environ["USE_MLX"] = "0"
```

Advanced Configuration
Advanced users can fine-tune the M3 optimizations by setting environment variables:

```bash
# Set MLX to use specific number of cores
export MLX_CORES=8

# Set memory limit (useful for smaller M3 chips)
export MLX_MEMORY_LIMIT=16GB

# Enable/disable Metal shader fusion
export MLX_METAL_SUBGRAPH_FUSION=1
```

These can also be set in your .env file. 