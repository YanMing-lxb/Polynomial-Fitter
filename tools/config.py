
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent  # 项目根目录
sys.path.append(str(BASE_DIR))  # 关键路径设置

from src.version import __version__  # noqa: E402

# -----------------------------------------------------------------------
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<| 项目配置 |>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# -----------------------------------------------------------------------

PROJECT_NAME = "Polynomial-Fitter"
ROOT_DIR = Path(__file__).parent.parent  # 设置项目根目录
SRC_DIR = ROOT_DIR / "src"
ENTRY_POINT = SRC_DIR / "main.py"
DATA_DIR = SRC_DIR / "assets"
ICON_FILE = DATA_DIR / "icon.ico"
VENV_NAME = ".venv"
__version__ = __version__

sys.path.append(str(ROOT_DIR))  # 关键路径设置
