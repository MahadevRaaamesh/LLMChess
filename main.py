import sys
import os

# Ensure the root directory is on the path so internal imports work correctly
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.app import run

if __name__ == "__main__":
    run()
