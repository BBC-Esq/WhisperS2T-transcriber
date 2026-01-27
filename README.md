<img width="757" height="490" alt="app_icon" src="https://github.com/user-attachments/assets/27c4f4c4-e947-4923-889c-63b246170c07" /><br>
> Uses the powerful WhisperS2T and Ctranslate2 libraries to batch transcribe multiple files

## Requirements
1) 🐍Python 3.11, 3.12 or 3.13
2) 📁[Git](https://git-scm.com/downloads)
3) 📁[Git Large File Storage](https://git-lfs.com/)
8) 🪟 Windows (linux not yet supported)

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
python install.py
```
# Usage
```
python main.py
```
The program will process any and all of the following file types:
* ```.aac```, ```.amr```, ```.asf```, ```.avi```, ```.flac```, ```.m4a```, ```.mkv```, ```.mp3```, ```.mp4```, ```.wav```, ```.webm```, ```.wma```.

### Important
All transcriptions are output in the same folder of the file that was transcribed.  If you'd like to change this behavior put an issue on Github requesting it.
