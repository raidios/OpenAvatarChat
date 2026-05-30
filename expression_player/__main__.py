"""支持两种方式：python3 -m expression_player（推荐）或 python3 expression_player/__main__.py"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    # 直接执行本文件时，把项目根目录加入 path，再按包名导入
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from expression_player.demo import main
else:
    from .demo import main

raise SystemExit(main())
