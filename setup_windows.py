import os
import sys
import tkinter
from tkinter import messagebox
import subprocess

def tkinter_message_box(title, message, type="info", yes_no=False):
    root = tkinter.Tk()
    root.withdraw()
    if type == "info":
        messagebox.showinfo(title, message)
    elif type == "error":
        messagebox.showerror(title, message)
    elif type == "yesno" and yes_no:
        return messagebox.askyesno(title, message)
    return None

def is_venv():
    if sys.prefix != sys.base_prefix:
        return True
    else:
        tkinter_message_box("No Virtual Environment", "Must create and activate a virtual environment before running this script.  Please see the instructions on the Github page for more details.", type="error")
        sys.exit(0)

def check_python_version_and_confirm():
    major, minor = map(int, sys.version.split()[0].split('.')[:2])
    if major < 3 or (major == 3 and minor < 10):
        tkinter_message_box("Python Version Error", "This program is currently only compatible with Python 3.10 or 3.11.", type="error")
        sys.exit(0)
    elif major == 3 and (minor == 12 or minor > 12):
        tkinter_message_box("Python Version Error", "Python 3.12+ detected. PyTorch is not currently compatible with Python 3.12 - exiting installer.", type="error")
        sys.exit(0)
    else:
        return True

def check_cuda_version():
    try:
        cuda_version_output = subprocess.check_output(["nvcc", "--version"]).decode('utf-8')
        if "release" in cuda_version_output:
            cuda_version = cuda_version_output.split("release ")[1].split(",")[0]
            major, minor = cuda_version.split('.')[:2]
            cuda_version_num = float(f"{major}.{minor}")
            if cuda_version_num < 12.1:
                tkinter_message_box("CUDA Check", f"Incorrect version of CUDA installed (Version: {cuda_version_num}). Please install CUDA 12.1+.  Exiting the installer.", type="error")
                sys.exit(0)
            else:
                tkinter_message_box("CUDA Check", f"CUDA version {cuda_version_num} detected. Proceeding with the GPU-accelerated installation.", type="info")
                return cuda_version_num
        else:
            tkinter_message_box("CUDA Check", "Unable to determine CUDA version. Exiting the installer.", type="error")
            sys.exit(0)
    except FileNotFoundError:
        tkinter_message_box("CUDA Check", "No CUDA installation detected. Exiting the installer.", type="error")
        sys.exit(0)

def install_pytorch(cuda_version_num):
    major, minor = map(int, sys.version.split()[0].split('.')[:2])
    if cuda_version_num >= 12.1:
        if minor == 11:
            os.system(f"{sys.executable} -m pip install https://download.pytorch.org/whl/cu121/torch-2.2.0%2Bcu121-cp311-cp311-win_amd64.whl#sha256=d79324159c622243429ec214a86b8613c1d7d46fc4821374d324800f1df6ade1")
            os.system(f"{sys.executable} -m pip install nvidia-ml-py==12.535.133")
        elif minor == 10:
            os.system(f"{sys.executable} -m pip install https://download.pytorch.org/whl/cu121/torch-2.2.0%2Bcu121-cp310-cp310-win_amd64.whl#sha256=8f54c647ee19c8b4c0aad158c73b83b2c06cb62351e9cfa981540ce7295a9015")
            os.system(f"{sys.executable} -m pip install nvidia-ml-py==12.535.133")
    else:
        if minor == 11:
            os.system(f"{sys.executable} -m pip install https://download.pytorch.org/whl/cpu/torch-2.2.0%2Bcpu-cp311-cp311-win_amd64.whl#sha256=58194066e594cd8aff27ddb746399d040900cc0e8a331d67ea98499777fa4d31")
        elif minor == 10:
            os.system(f"{sys.executable} -m pip install https://download.pytorch.org/whl/cpu/torch-2.2.0%2Bcpu-cp310-cp310-win_amd64.whl#sha256=15a657038eea92ac5db6ab97b30bd4b5345741b49553b2a7e552e80001297124")

def main():
    is_venv()
    check_python_version_and_confirm()
    cuda_version_num = check_cuda_version()
    os.system(f"{sys.executable} -m pip install --upgrade pip")
    os.system(f"{sys.executable} -m pip install -r requirements.txt")
    install_pytorch(cuda_version_num)
    os.system(f"{sys.executable} -m pip install -U --no-deps git+https://github.com/shashikg/WhisperS2T.git")
    print("\033[92mInstallation successful. Run 'python whispers2t_batch_gui.py' to start!\033[0m")

if __name__ == "__main__":
    main()
