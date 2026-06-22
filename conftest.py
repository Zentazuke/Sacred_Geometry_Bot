import sys
from pathlib import Path

# make `import src...` work from the project root during tests
sys.path.insert(0, str(Path(__file__).resolve().parent))
