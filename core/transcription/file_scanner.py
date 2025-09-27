"""File discovery and scanning utilities."""
from pathlib import Path
from typing import List

class FileScanner:
    """Scans directories for audio files."""
    
    def scan_directory(self, directory: Path, extensions: List[str], 
                       recursive: bool = False) -> List[Path]:
        """Scan directory for files matching extensions."""
        files = []
        patterns = [f'*{ext}' for ext in extensions]
        
        for pattern in patterns:
            if recursive:
                files.extend(directory.rglob(pattern))
            else:
                files.extend(directory.glob(pattern))
                
        return sorted(files)
    
    def count_files(self, directory: Path, extensions: List[str],
                   recursive: bool = False) -> int:
        """Count files without collecting them."""
        return len(self.scan_directory(directory, extensions, recursive))