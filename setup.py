import subprocess
import sys
import os
import time
from pathlib import Path

def install_libraries_with_retry(max_retries=3, delay=3):
    libraries = [
        # "accelerate==0.33.0",
        "certifi==2024.7.4",
        "charset-normalizer==3.3.2",
        "colorama==0.4.6",
        "ctranslate2==4.3.1",
        "filelock==3.15.4",
        "fsspec==2024.5.0",
        "huggingface-hub==0.24.1",
        "idna==3.7",
        "jinja2==3.1.4",
        "llvmlite==0.43.0",
        "markdown-it-py==3.0.0",
        "MarkupSafe==2.1.5",
        "mdurl==0.1.2",
        "more-itertools==10.3.0",
        "mpmath==1.3.0",
        "networkx==3.3",
        "numba==0.60.0",
        "numpy==1.26.4",
        "nvidia-cublas-cu12==12.1.3.1",
        "nvidia-cuda-runtime-cu12==12.1.105",
        "nvidia-ml-py==12.555.43",
        "openai-whisper==20231117",
        # "optimum==1.21.2",
        "packaging==24.1",
        "platformdirs==4.2.2",
        "psutil==6.0.0",
        "pygments==2.18.0",
        "pyyaml==6.0.1",
        "regex==2024.5.15",
        "requests==2.32.3",
        "rich==13.7.1",
        "safetensors==0.4.3",
        "sympy==1.12.1",
        "tiktoken==0.7.0",
        "https://download.pytorch.org/whl/cu121/torch-2.2.2%2Bcu121-cp311-cp311-win_amd64.whl#sha256=efbcfdd4399197d06b32f7c0e1711c615188cdd65427b933648c7478fb880b3f",
        "tokenizers==0.19.1",
        "tqdm==4.66.4",
        "typing-extensions==4.12.2",
        "typing_extensions==4.12.2",
        "urllib3==2.2.2",
        'git+https://github.com/shashikg/WhisperS2T.git@e7f7e6dbfdc7f3a39454feb9dd262fd3653add8c#egg=whisper_s2t'
    ]

    failed_installations = []
    multiple_attempts = []

    for library in libraries:
        for attempt in range(max_retries):
            try:
                print(f"\nAttempt {attempt + 1} of {max_retries}: Installing {library}")
                command = [sys.executable, "-m", "uv", "pip", "install", library, "--no-deps", "--no-cache-dir"]
                subprocess.run(command, check=True, capture_output=True, text=True)
                print(f"Successfully installed {library}")
                if attempt > 0:
                    multiple_attempts.append((library, attempt + 1))
                break
            except subprocess.CalledProcessError as e:
                print(f"Attempt {attempt + 1} failed. Error: {e.stderr.strip()}")
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
    
    # install uv
    print("\033[92mInstalling uv:\033[0m")
    subprocess.run(["pip", "install", "uv"], check=True)

    # install pyside6
    print("\033[92mInstalling PySide6:\033[0m")
    subprocess.run(["uv", "pip", "install", "pyside6", "--no-cache-dir", "--link-mode=copy"], check=True)
    
    # Upgrade pip, setuptools, and wheel using uv
    print("\033[92mUpgrading pip, setuptools, and wheel:\033[0m")
    subprocess.run(f"{sys.executable} -m uv pip install --upgrade pip setuptools wheel", shell=True, check=True)
    
    # install other libraries
    print("\033[92mInstalling dependencies:\033[0m")
    failed, multiple = install_libraries_with_retry()
    
    if not failed:
        print("\033[92mInstallation was successful! The program is ready to use.")
        print(f"To run it, enter the command: python ct2_main.py\033[0m")
    else:
        print("\033[91mInstallation encountered some issues. Please review the installation summary above.\033[0m")

    end_time = time.time()
    total_time = end_time - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)

    print(f"\033[92m\nTotal installation time: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}\033[0m")

if __name__ == "__main__":
    main()