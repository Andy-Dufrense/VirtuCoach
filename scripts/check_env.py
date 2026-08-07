#!/usr/bin/env python
"""VirtuCoach 环境诊断工具。

用法: python check_env.py

检查清单:
- Python 版本和路径
- E:\Lib\site-packages 可用性
- 关键依赖版本
- 端口 7160 是否空闲
- .env 文件及必需 API Key
- reference_hands 数据库状态
- knowledge/ 目录完整性
- starlette 版本兼容性
"""

import os
import sys
import socket
import importlib
from pathlib import Path

# Windows: force UTF-8 output
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

PROJECT_ROOT = Path(__file__).parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
SITE_PACKAGES = Path("E:/Lib/site-packages")

CHECKS_PASSED = 0
CHECKS_FAILED = 0


def ok(msg: str):
    global CHECKS_PASSED
    CHECKS_PASSED += 1
    print(f"  [PASS] {msg}")


def warn(msg: str):
    global CHECKS_FAILED
    CHECKS_FAILED += 1
    print(f"  [WARN] {msg}")


def fail(msg: str):
    global CHECKS_FAILED
    CHECKS_FAILED += 1
    print(f"  [FAIL] {msg}")


def check_python():
    print("\n[Python]")
    v = sys.version_info
    if (v.major, v.minor) == (3, 10):
        ok(f"Python {v.major}.{v.minor}.{v.micro} ({sys.executable})")
    else:
        warn(f"Python {v.major}.{v.minor}.{v.micro} — 建议 3.10 ({sys.executable})")


def check_site_packages():
    print("\n[E:\\Lib\\site-packages]")
    if SITE_PACKAGES.is_dir():
        ok(f"目录存在: {SITE_PACKAGES}")
    else:
        fail(f"目录不存在: {SITE_PACKAGES}")
        return

    required = {
        "mediapipe": "0.10.8",
        "fastapi": "0.104.1",
        "uvicorn": None,
        "chromadb": None,
        "PIL": None,
        "cv2": None,
        "numpy": None,
        "sounddevice": None,
    }
    for name, min_ver in required.items():
        try:
            mod = importlib.import_module(name)
            ver = getattr(mod, "__version__", "?")
            if min_ver and ver < min_ver:
                warn(f"{name}=={ver} (需要 >={min_ver})")
            else:
                ok(f"{name}=={ver}")
        except ImportError:
            fail(f"{name} 未安装")


def check_port():
    print("\n[端口 7160]")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 7160))
        sock.close()
        ok("端口 7160 空闲")
    except OSError:
        fail("端口 7160 已被占用")


def check_env_file():
    print("\n[.env]")
    env_path = PROJECT_ROOT / ".env"
    if not env_path.is_file():
        fail(".env 文件不存在")
        return

    ok(f".env 存在")
    content = env_path.read_text(encoding="utf-8")
    for key in ["DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY"]:
        if key in content:
            ok(f"{key} 已配置")
        else:
            warn(f"{key} 未找到")


def check_reference_db():
    print("\n[参考图数据库]")
    db_path = BACKEND_DIR / "reference_hands" / "references.db"
    if not db_path.is_file():
        fail(f"数据库不存在: {db_path}")
        return

    import sqlite3
    try:
        conn = sqlite3.connect(str(db_path))
        count = conn.execute("SELECT COUNT(*) FROM hand_references").fetchone()[0]
        conn.close()
        ok(f"数据库可读，{count} 条记录")
    except Exception as e:
        fail(f"数据库读取失败: {e}")


def check_knowledge():
    print("\n[knowledge/ 目录]")
    kb_dir = PROJECT_ROOT / "knowledge"
    if not kb_dir.is_dir():
        fail(f"目录不存在: {kb_dir}")
        return

    chords = sorted((kb_dir / "chords").glob("*.md"))
    problems = sorted((kb_dir / "problems").glob("*.md"))
    techniques = list((kb_dir / "techniques").glob("**/*.md"))

    ok(f"和弦: {len(chords)} 个")
    ok(f"问题: {len(problems)} 个")
    ok(f"技巧: {len(techniques)} 个")
    ok(f"总计: {len(chords) + len(problems) + len(techniques)} 个 KB 文件")


def check_starlette():
    print("\n[starlette 兼容性]")
    try:
        import starlette
        ver = starlette.__version__
        if ver.startswith("0.27"):
            ok(f"starlette=={ver} (兼容 FastAPI 0.104.1)")
        else:
            fail(f"starlette=={ver} (需要 0.27.0，FastAPI 0.104.1 不兼容 >=1.x)")
            print("  修复: pip install starlette==0.27.0 --target E:/Lib/site-packages --force-reinstall --no-deps")
    except ImportError:
        fail("starlette 未安装")


def main():
    print("=" * 50)
    print("  VirtuCoach 环境诊断")
    print("=" * 50)

    check_python()
    check_site_packages()
    check_port()
    check_env_file()
    check_reference_db()
    check_knowledge()
    check_starlette()

    print("\n" + "=" * 50)
    total = CHECKS_PASSED + CHECKS_FAILED
    print(f"  结果: {CHECKS_PASSED} 通过, {CHECKS_FAILED} 失败 (共 {total} 项)")
    if CHECKS_FAILED == 0:
        print("  所有检查通过，可以启动服务器。")
    else:
        print("  请先修复上面的失败项再启动。")
    print("=" * 50)

    return CHECKS_FAILED == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
