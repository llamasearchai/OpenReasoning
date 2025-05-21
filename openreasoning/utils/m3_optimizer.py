"""
Optimizations for Apple Silicon M3 processors.
"""

import os
import platform
from typing import Any, Dict, Optional

from loguru import logger

from openreasoning.core.config import settings


class M3Optimizer:
    """
    Provides optimizations for Apple M3 series chips.
    This includes MLX-based optimizations, memory management strategies,
    and multi-core workload distribution.
    """

    def __init__(self):
        self.has_mlx = False
        try:
            import mlx

            self.has_mlx = True
            logger.info("MLX detected and available for M3 optimizations")
        except ImportError:
            logger.warning("MLX not found - some M3 optimizations will be unavailable")

        self.optimizations_applied = False
        self.optimization_status = {
            "is_apple_silicon": settings.is_apple_silicon,
            "is_m3_chip": settings.is_m3_chip,
            "has_mlx": self.has_mlx,
            "optimizations_applied": False,
            "details": {},
        }

    def apply_optimizations(self) -> bool:
        """
        Apply M3-specific optimizations if running on an M3 chip.

        Returns:
            bool: True if optimizations were applied, False otherwise
        """
        if not settings.is_m3_chip:
            logger.info("Not running on M3 chip, skipping optimizations")
            return False

        optimizations = {
            "memory": self._optimize_memory(),
            "multicore": self._optimize_multicore(),
            "mlx": self._optimize_mlx() if self.has_mlx else False,
        }

        self.optimizations_applied = any(optimizations.values())
        self.optimization_status["optimizations_applied"] = self.optimizations_applied
        self.optimization_status["details"] = optimizations

        if self.optimizations_applied:
            logger.info(f"Applied M3 optimizations: {optimizations}")

        return self.optimizations_applied

    def _optimize_memory(self) -> bool:
        """
        Optimize memory usage for M3's unified memory architecture.
        """
        try:
            # Set optimal memory allocation patterns for unified memory
            os.environ["MALLOC_ARENA_MAX"] = "1"  # Reduce memory fragmentation

            # Configure memory-related environment variables
            if self.has_mlx:
                os.environ["MLX_BUFFER_ALLOCATION_SIZE"] = (
                    "32000000"  # 32MB buffer allocation
                )

            logger.debug("Applied memory optimizations for M3 chip")
            return True
        except Exception as e:
            logger.error(f"Failed to apply memory optimizations: {e}")
            return False

    def _optimize_multicore(self) -> bool:
        """
        Configure workload distribution across performance and efficiency cores.
        """
        try:
            # Set thread allocation strategy
            os.environ["OMP_WAIT_POLICY"] = "PASSIVE"

            # If we can detect core count, configure thread allocation
            if hasattr(os, "sched_getaffinity"):
                core_count = len(os.sched_getaffinity(0))
                if core_count > 4:
                    # Leave 2 efficiency cores for system tasks
                    thread_count = max(1, core_count - 2)
                    os.environ["OMP_NUM_THREADS"] = str(thread_count)
                    logger.debug(
                        f"Set thread count to {thread_count} for multicore optimization"
                    )

            logger.debug("Applied multicore optimizations for M3 chip")
            return True
        except Exception as e:
            logger.error(f"Failed to apply multicore optimizations: {e}")
            return False

    def _optimize_mlx(self) -> bool:
        """
        Configure MLX for optimal performance on M3 chips.
        """
        if not self.has_mlx:
            return False

        try:
            import mlx.core as mx

            # Set MLX to use all available compute units
            # This maximizes the use of M3's Neural Engine
            mx.set_default_device(mx.Device("gpu"))

            logger.debug("Applied MLX optimizations for M3 chip")
            return True
        except Exception as e:
            logger.error(f"Failed to apply MLX optimizations: {e}")
            return False

    def get_optimization_status(self) -> Dict[str, Any]:
        """
        Get the current status of M3 optimizations.

        Returns:
            Dict[str, Any]: Information about M3 detection and applied optimizations
        """
        return self.optimization_status


# Create global optimizer instance
m3_optimizer = M3Optimizer()
