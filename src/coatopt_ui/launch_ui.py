import os
import sys
import socket
import webbrowser
from threading import Timer
from pathlib import Path

def find_free_port() -> int:
    """Find an available port on localhost."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()
    return port

def open_browser(port: int):
    """Open default web browser to local app URL."""
    webbrowser.open(f"http://127.0.0.1:{port}")

def main():
    # Resolve project root and source path
    current_dir = Path(__file__).resolve().parent
    src_dir = current_dir.parent
    project_root = src_dir.parent
    
    # Add src_dir to sys.path so we can import coatopt_ui.app
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
        
    # Also add CoatingAnalysis path if it's not already in sys.path
    lib_path = "/Users/simon/Library/CloudStorage/GoogleDrive-simon.tait@ligo.org/My Drive/BackupFromDropbox/Python/CoatingAnalysis/src"
    if lib_path not in sys.path:
        sys.path.insert(0, lib_path)
        
    # Print welcome message
    print("======================================================")
    print("      INTERACTIVE COATING STACK SIMULATOR UI")
    print("======================================================")
    print(f"Project path: {project_root}")
    print(f"CoatingAnalysis path: {lib_path}\n")

    port = find_free_port()
    print(f"Starting local server on http://127.0.0.1:{port} ...")
    
    # Set timer to open browser once server is running
    Timer(1.5, open_browser, args=(port,)).start()
    
    # Run uvicorn
    import uvicorn
    uvicorn.run("coatopt_ui.app:app", host="127.0.0.1", port=port, reload=False)

if __name__ == "__main__":
    main()
