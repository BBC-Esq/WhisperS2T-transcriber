# 🚀WhisperS2T-transcriber🚀
* Uses the powerful WhisperS2T and Ctranslate2 libraries to batch transcribe multiple files

## Requirements
1) 🐍[Python 3.11](https://www.python.org/downloads/release/python-3117/)
2) 📁[Git](https://git-scm.com/downloads)
3) 📁[Git Large File Storage](https://git-lfs.com/)
8) 🪟 Windows (linux not yet supported)
   > You can modify the installation instructions manually for Linux, I just don't have Linux in order to test them reliably.  You can do this by analyzing the ```setup_gpu.py``` or ```setup_cpu_only.py``` file to see what libraries are required.

# Installation
Download the latest release and extract the files your computer.  Navigate to the respository folder, create a command prompt, and run the following commands:

```
python -m venv .
```
```
.\Scripts\activate
```
  > Run this again to activate the environment each time you restart the program.

Run the installation script:
```
python setup.py
```
# Usage
```
python whispers2t_batch_gui.py
```
The program will process any and all of the following file types:
* ```.aac```, ```.amr```, ```.asf```, ```.avi```, ```.flac```, ```.m4a```, ```.mkv```, ```.mp3```, ```.mp4```, ```.wav```, ```.webm```, ```.wma```.

### Important
All transcriptions are output in the same folder of the file that was transcribed.  If you'd like to change this behavior put an issue on Github requesting it.
