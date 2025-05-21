"""
Unit tests for M3 optimizer.
"""

import os
import platform
from unittest.mock import MagicMock, patch

import pytest

from openreasoning.utils.m3_optimizer import M3Optimizer


class TestM3Optimizer:
    """Tests for M3 optimizer."""

    @patch("platform.machine")
    @patch("platform.system")
    @patch("openreasoning.utils.m3_optimizer.os.popen")
    def test_check_m3(self, mock_popen, mock_system, mock_machine):
        """Test M3 detection."""
        # Mock Apple Silicon
        mock_machine.return_value = "arm64"
        mock_system.return_value = "Darwin"

        # Mock M3 chip detection
        mock_process = MagicMock()
        mock_process.read.return_value = "Apple M3 Max"
        mock_popen.return_value = mock_process

        optimizer = M3Optimizer()
        assert optimizer._check_m3() is True

        # Change to non-M3 chip
        mock_process.read.return_value = "Apple M1 Pro"
        assert optimizer._check_m3() is False

        # Change to non-Apple Silicon
        mock_machine.return_value = "x86_64"
        assert optimizer._check_m3() is False

    @patch("openreasoning.utils.m3_optimizer.M3Optimizer._check_m3")
    @patch("openreasoning.utils.m3_optimizer.M3Optimizer._get_device_info")
    @patch("os.environ")
    def test_apply_optimizations(
        self, mock_environ, mock_get_device_info, mock_check_m3
    ):
        """Test optimization application."""
        # Mock M3 chip
        mock_check_m3.return_value = True
        mock_get_device_info.return_value = {
            "chip_variant": "M3 Max",
            "total_ram_gb": 64,
            "performance_cores": 10,
            "efficiency_cores": 4,
        }

        # Mock environment variables
        mock_environ.__setitem__ = MagicMock()

        # Test optimization application with MLX available
        with patch.dict("sys.modules", {"mlx": MagicMock(), "mlx.core": MagicMock()}):
            optimizer = M3Optimizer()
            result = optimizer.apply_optimizations()

            assert result is True
            assert optimizer.optimizations_applied is True

            # Check that MLX settings were applied
            mock_environ.__setitem__.assert_any_call("MLX_CORES", "12")
            mock_environ.__setitem__.assert_any_call("MLX_METAL_SUBGRAPH_FUSION", "1")

        # Test optimization application without MLX
        with patch.dict("sys.modules", {"mlx": None}):
            mock_environ.clear_mock()
            optimizer = M3Optimizer()
            result = optimizer.apply_optimizations()

            # Should still return True but not set MLX variables
            assert result is True
            assert optimizer.optimizations_applied is True

    def test_get_optimization_status(self):
        """Test getting optimization status."""
        optimizer = M3Optimizer()

        # Basic status check
        status = optimizer.get_optimization_status()
        assert "is_m3" in status
        assert "optimizations_applied" in status
        assert "device_info" in status
