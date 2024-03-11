# 🚀WhisperS2T-transcriber🚀
* Uses the powerful WhisperS2T and Ctranslate2 libraries to batch transcribe multiple files
* THE fastest (while still maintaining quality) Whisper model transcriber available.

## Requirements
1) 🐍[Python 3.10](https://www.python.org/downloads/release/python-31011/) or [Python 3.11](https://www.python.org/downloads/release/python-3117/)
2) 📁[Git](https://git-scm.com/downloads)
3) 📁[Git Large File Storage](https://git-lfs.com/).
6) 🟢[CUDA 12.1+](https://developer.nvidia.com/cuda-toolkit) for Nvidia GPU acceleration.
   > AMD acceleration not yet supported.
8) 🪟 Windows
   > You can modify the installation instructions manually for Linux, I just don't have Linux in order to test them reliably.  You can do this by analyzing the ```requirements.txt``` and ```setup_windows.py``` files to see what the libraries are required.

# Installation
Download the latest release and extract the files your computer.  Navigate to the respository folder, create a command prompt, and run the following commands:

```
python -m venv .
```
```
.\Scripts\activate
```
  > Run this again to activate the environment each time you restart the program.
```
python setup_windows.py
```
# Usage
```
python whispers2t_batch_gui.py
```
The program will process any and all of the following file types:
* ```.mp3```, ```.wav```, ```.flac```, ```.wma```, ```.aac```, ```.m4a```, ```.avi```, ```.mkv```, ```.mp4```, ```.asf```, ```.amr```.
