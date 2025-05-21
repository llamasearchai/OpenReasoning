#!/usr/bin/env python3
"""
Script to test OpenReasoning installation.
"""
import os
import sys
import platform

def check_import():
    try:
        import openreasoning
        print(f"OpenReasoning version: {openreasoning.__version__}")
        return True
    except ImportError as e:
        print(f"Error importing OpenReasoning: {e}")
        return False

def check_modules():
    modules = [
        "openreasoning.core",
        "openreasoning.models",
        "openreasoning.retrieval",
        "openreasoning.agents",
        "openreasoning.training",
        "openreasoning.api",
        "openreasoning.cli",
        "openreasoning.utils"
    ]

    success = True
    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module}: {e}")
            success = False
    return success

def check_m3_detection():
    """Check M3 detection functionality."""
    try:
        from openreasoning.core.config import settings
        from openreasoning.utils.m3_optimizer import m3_optimizer

        print(f"\nM3 Detection:")
        print(f"Is Apple Silicon: {settings.is_apple_silicon}")
        print(f"Is M3 chip: {settings.is_m3_chip}")
        
        if settings.is_m3_chip:
            print("\nApplying M3 optimizations...")
            optimized = m3_optimizer.apply_optimizations()
            print(f"Optimizations applied: {optimized}")
            
            print("\nOptimization status:")
            status = m3_optimizer.get_optimization_status()
            for key, value in status.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for subkey, subvalue in value.items():
                        print(f"    {subkey}: {subvalue}")
                else:
                    print(f"  {key}: {value}")
        
        return True
    except Exception as e:
        print(f"Error checking M3 detection: {e}")
        return False

def main():
    print("Testing OpenReasoning installation...\n")

    import_ok = check_import()
    if not import_ok:
        print("\nFailed to import OpenReasoning. Make sure it's installed correctly.")
        sys.exit(1)

    print("\nChecking modules:")
    modules_ok = check_modules()

    if modules_ok:
        print("\nAll modules imported successfully.")
    else:
        print("\nSome modules failed to import.")

    # Check M3 detection
    if platform.machine() == "arm64" and platform.system() == "Darwin":
        m3_ok = check_m3_detection()
        if not m3_ok:
            print("\nM3 detection checks failed.")

    # Check for API keys
    print("\nChecking API keys:")
    for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "HUGGINGFACE_API_KEY"]:
        if key in os.environ:
            print(f"✓ {key} is set")
        else:
            print(f"✗ {key} is not set")

    # Test CLI
    try:
        import openreasoning.cli
        print("\n✓ CLI module is available")
    except ImportError as e:
        print(f"\n✗ CLI module not available: {e}")

    print("\nInstallation test complete.")

if __name__ == "__main__":
    main() 