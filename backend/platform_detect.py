"""
Platform detection for backend selection.
"""

import platform
import subprocess
from typing import Literal, Optional, Tuple


def is_apple_silicon() -> bool:
    """
    Check if running on Apple Silicon (arm64 macOS).
    
    Returns:
        True if on Apple Silicon, False otherwise
    """
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def get_mac_model_identifier() -> Optional[str]:
    """
    Get the Mac model identifier (e.g., 'Mac16,10' for Mac Mini M4).
    
    Returns:
        Model identifier string or None if not on macOS or detection fails
    """
    if platform.system() != "Darwin":
        return None
    
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.model"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    
    return None


def get_system_memory_gb() -> Optional[float]:
    """
    Get total system memory in GB.
    
    Returns:
        Total RAM in GB or None if detection fails
    """
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                bytes_ram = int(result.stdout.strip())
                return bytes_ram / (1024 ** 3)
        except Exception:
            pass
    
    # Fallback for other platforms
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        pass
    
    return None


def get_optimal_whisper_model() -> str:
    """
    Determine the optimal Whisper model size based on hardware.
    
    Returns:
        Recommended model size string
    """
    model_id = get_mac_model_identifier()
    ram_gb = get_system_memory_gb()
    
    # Mac Mini M4 (Mac16,10) with 16GB+ RAM -> use large-v3-turbo
    if model_id == "Mac16,10" and ram_gb is not None and ram_gb >= 16:
        print(f"Detected Mac Mini M4 with {ram_gb:.1f}GB RAM, using large-v3-turbo")
        return "large-v3-turbo"
    
    # Default fallback
    return "base"


def get_backend_type() -> Literal["mlx", "pytorch"]:
    """
    Detect the best backend for the current platform.
    
    Returns:
        "mlx" on Apple Silicon (if MLX is available), "pytorch" otherwise
    """
    if is_apple_silicon():
        try:
            import mlx
            return "mlx"
        except ImportError:
            # MLX not installed, fallback to PyTorch
            return "pytorch"
    return "pytorch"
