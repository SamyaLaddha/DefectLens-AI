import sys
import os

# Add backend directory to sys.path to enable direct importing of the app package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
