import sys
import subprocess
import time

def has_nvidia_gpu():
    """
    Check if an NVIDIA GPU is present by attempting to run 'nvidia-smi'.
    Returns True if detected, otherwise False.
    """
    try:
        result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def get_python_version_key():
    """
    Returns a key for the current Python version in the form 'cp<major><minor>'.
    For example, Python 3.11 returns 'cp311' and Python 3.12 returns 'cp312'.
    """
    return f"cp{sys.version_info.major}{sys.version_info.minor}"

# 1. Dependencies that are installed regardless of hardware or Python version.
common_libraries = [
    "certifi==2025.1.31",
    "charset-normalizer==3.4.1",
    "colorama==0.4.6",
    "ctranslate2==4.5.0",
    "filelock==3.17.0",
    "fsspec==2024.9.0",
    "huggingface-hub==0.29.3",
    "idna==3.10",
    "Jinja2==3.1.6",
    "llvmlite==0.44.0",
    "markdown-it-py==3.0.0",
    "MarkupSafe==3.0.2",
    "mdurl==0.1.2",
    "more-itertools==10.6.0",
    "mpmath==1.3.0",
    "networkx==3.4.2",
    "numba==0.61.0",
    "numpy==1.26.4",
    "openai-whisper==20240930",
    "packaging==24.2",
    "platformdirs==4.3.6",
    "psutil==7.0.0",
    "Pygments==2.19.1",
    "PyYAML==6.0.2",
    "regex==2024.11.6",
    "requests==2.32.3",
    "rich==13.9.4",
    "safetensors==0.5.3",
    "sympy==1.13.1",
    "tiktoken==0.9.0",
    "tokenizers==0.21.0",
    "tqdm==4.67.1",
    "typing_extensions==4.12.2",
    "urllib3==2.3.0",
    'git+https://github.com/shashikg/WhisperS2T.git@e7f7e6dbfdc7f3a39454feb9dd262fd3653add8c'
]

# 2. GPU-only dependencies that vary by Python version.
gpu_libs = {
    "cp311": [
        "nvidia-cublas-cu12==12.4.5.8",
        "nvidia-cuda-runtime-cu12==12.4.127",
        "nvidia-cudnn-cu12==9.1.0.70",
        "nvidia-ml-py==12.570.86"
    ],
    "cp312": [
        # Update these placeholders with the correct versions for Python 3.12
        "nvidia-cublas-cu12==12.4.5.8",
        "nvidia-cuda-runtime-cu12==12.4.127",
        "nvidia-cudnn-cu12==9.1.0.70",
        "nvidia-ml-py==12.570.86"
    ]
}

# 3. CPU-only dependencies that vary by Python version.
# (Currently, there are no additional CPU-only dependencies beyond common libraries.
#  Place any CPU-only libraries here if needed.)
cpu_libs = {
    "cp311": [],
    "cp312": []
}

# 4. PyTorch wheel URLs differ for GPU and CPU installations and may vary by Python version.
torch_libs = {
    "cp311": {
        "GPU": "https://download.pytorch.org/whl/cu124/torch-2.6.0%2Bcu124-cp311-cp311-win_amd64.whl#sha256=6a1fb2714e9323f11edb6e8abf7aad5f79e45ad25c081cde87681a18d99c29eb",
        "CPU": "https://download.pytorch.org/whl/cpu/torch-2.6.0%2Bcpu-cp311-cp311-win_amd64.whl#sha256=24c9d3d13b9ea769dd7bd5c11cfa1fc463fd7391397156565484565ca685d908"
    },
    "cp312": {
        "GPU": "https://download.pytorch.org/whl/cu124/torch-2.6.0%2Bcu124-cp312-cp312-win_amd64.whl#sha256=3313061c1fec4c7310cf47944e84513dcd27b6173b72a349bb7ca68d0ee6e9c0",
        "CPU": "https://download.pytorch.org/whl/cpu/torch-2.6.0%2Bcpu-cp312-cp312-win_amd64.whl#sha256=4027d982eb2781c93825ab9527f17fbbb12dbabf422298e4b954be60016f87d8"
    }
}

def install_libraries_with_retry(libraries, max_retries=3, delay=3):
    """
    Install each library using pip with retry logic.
    Returns a tuple: (failed_installations, multiple_attempts).
    """
    failed_installations = []
    multiple_attempts = []
    for library in libraries:
        for attempt in range(max_retries):
            try:
                print(f"\nAttempt {attempt + 1} of {max_retries}: Installing {library}")
                command = [sys.executable, "-m", "uv", "pip", "install", library, "--no-deps"]
                subprocess.run(command, check=True, capture_output=True, text=True)
                print(f"Successfully installed {library}")
                if attempt > 0:
                    multiple_attempts.append((library, attempt + 1))
                break
            except subprocess.CalledProcessError as e:
                print(f"Attempt {attempt + 1} failed for {library}. Error: {e.stderr.strip()}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print(f"Failed to install {library} after {max_retries} attempts.")
                    failed_installations.append(library)
    print("\n--- Installation Summary ---")
    if failed_installations:
        print("\nThe following libraries failed to install:")
        for lib in failed_installations:
            print(f"- {lib}")
    if multiple_attempts:
        print("\nThe following libraries required multiple attempts to install:")
        for lib, attempts in multiple_attempts:
            print(f"- {lib} (took {attempts} attempts)")
    if not failed_installations and not multiple_attempts:
        print("\nAll libraries installed successfully on the first attempt.")
    elif not failed_installations:
        print("\nAll libraries were eventually installed successfully.")
    return failed_installations, multiple_attempts

def main():
    start_time = time.time()

    python_key = get_python_version_key()
    if python_key not in torch_libs:
        print(f"Unsupported Python version: {python_key}. This installer supports Python 3.11 and 3.12.")
        sys.exit(1)

    hardware = "GPU" if has_nvidia_gpu() else "CPU"
    print(f"Detected Python version: {python_key}, Hardware: {hardware}")

    # Build final dependency list:
    # Start with common libraries.
    libraries = list(common_libraries)
    # Append additional libraries based on hardware.
    if hardware == "GPU":
        libraries += gpu_libs.get(python_key, [])
        torch_wheel = torch_libs[python_key]["GPU"]
    else:
        libraries += cpu_libs.get(python_key, [])
        torch_wheel = torch_libs[python_key]["CPU"]
    # Append the appropriate PyTorch wheel.
    libraries.append(torch_wheel)

    # Install uv package.
    print("\033[92mInstalling uv:\033[0m")
    subprocess.run([sys.executable, "-m", "pip", "install", "uv"], check=True)

    # Install PySide6.
    print("\033[92mInstalling PySide6:\033[0m")
    subprocess.run([sys.executable, "-m", "uv", "pip", "install", "pyside6", "--link-mode=copy"], check=True)

    # Upgrade pip, setuptools, and wheel using uv.
    print("\033[92mUpgrading pip, setuptools, and wheel:\033[0m")
    upgrade_command = f"{sys.executable} -m uv pip install --upgrade pip setuptools wheel"
    subprocess.run(upgrade_command, shell=True, check=True)

    # Install selected dependencies with retry logic.
    print("\033[92mInstalling dependencies:\033[0m")
    failed, multiple = install_libraries_with_retry(libraries)

    if not failed:
        print("\033[92mInstallation was successful! The program is ready to use.\033[0m")
        print("To run it, enter the command: python whispers2t_batch_gui.py")
    else:
        print("\033[91mInstallation encountered some issues. Please review the installation summary above.\033[0m")

    end_time = time.time()
    total_time = end_time - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\033[92m\nTotal installation time: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}\033[0m")

if __name__ == "__main__":
    main()
