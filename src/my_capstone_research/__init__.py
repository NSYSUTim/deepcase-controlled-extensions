"""研究版 DeepCASE 擴充套件。

這個 package 與現有 `src/my_capstone/` baseline 並列存在。
原則是：

- baseline 不動
- 每個研究題目各自獨立
- 共用研究工具放在 `common/`
- 研究版 CLI 由 `main_research.py` 進入
"""

from .registry import get_research_pipeline, load_research_config

__all__ = ["get_research_pipeline", "load_research_config"]
