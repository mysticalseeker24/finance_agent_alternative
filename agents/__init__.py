"""Agent module initialization."""

import sys
from pathlib import Path

# Add the project root directory to Python path for consistent module resolution
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
