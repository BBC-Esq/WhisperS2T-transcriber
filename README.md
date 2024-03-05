# 🚀WhisperS2T-transcriber🚀
* Uses the powerful WhisperS2T and Ctranslate2 libraries to batch transcribe multiple files
* THE fastest (while still maintaining quality) Whisper model transcriber available.

## Requirements
* Python 3.11
* Nvidia GPU with CUDA 12+ software installed
  > For the initial releases I'm requiring this, but in the future I plan on support cpu-only inatallations and/or AMD gpu-acceleration as well.
* Also, you might have to customize the instructions for Linux because I don't have Linux to test.

## Installation Instructions
Download the latest release and extract the files your computer.<br><br>  Navigate to the respository folder and create a command prompt and run:

### Create a virtual environment
```
python -m venv .
```
### Activate virtual environment
```
.\Scripts\activate
```
  > Run this again to activate the environment each time you restart the program.

### Upgrade pip
```
python -m pip install --upgrade pip
```
### Install Requirements
```
pip3 install -r requirements.txt
```
### Install Pytorch
```
pip3 install https://download.pytorch.org/whl/cu121/torch-2.2.0%2Bcu121-cp311-cp311-win_amd64.whl#sha256=d79324159c622243429ec214a86b8613c1d7d46fc4821374d324800f1df6ade1
```
  > If using Python 3.11
```
pip3 install https://download.pytorch.org/whl/cu121/torch-2.2.0%2Bcu121-cp310-cp310-win_amd64.whl#sha256=8f54c647ee19c8b4c0aad158c73b83b2c06cb62351e9cfa981540ce7295a9015
```
  > If using Pyton 3.10
### Install WhisperS2T
```
pip3 install -U --no-deps git+https://github.com/shashikg/WhisperS2T.git
```

## Usage
```
python whispers2t_batch_gui.py
```
## No GUI
```
python whispers2t_batch.py
```
