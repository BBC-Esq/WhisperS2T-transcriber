"""CUDA path configuration utilities."""
import os
import sys
from pathlib import Path

def setup_cuda_paths():
    """Configure CUDA paths for Windows environments."""
    try:
        # Get venv base path
        venv_base = Path(sys.executable).parent.parent
        nvidia_base_path = venv_base / 'Lib' / 'site-packages' / 'nvidia'
        
        # Define CUDA paths
        cuda_paths = [
            nvidia_base_path / 'cuda_runtime' / 'bin',
            nvidia_base_path / 'cuda_runtime' / 'lib' / 'x64',
            nvidia_base_path / 'cuda_runtime' / 'include',
            nvidia_base_path / 'cublas' / 'bin',
            nvidia_base_path / 'cudnn' / 'bin',
            nvidia_base_path / 'cuda_nvrtc' / 'bin',
        ]
        
        # Convert to strings and filter existing paths
        paths_to_add = [str(p) for p in cuda_paths if p.exists()]
        
        if paths_to_add:
            # Update PATH environment variable
            current_path = os.environ.get('PATH', '')
            new_paths = os.pathsep.join(paths_to_add)
            
            if current_path:
                os.environ['PATH'] = f"{new_paths}{os.pathsep}{current_path}"
            else:
                os.environ['PATH'] = new_paths
                
    except Exception as e:
        print(f"Warning: Could not setup CUDA paths: {e}")