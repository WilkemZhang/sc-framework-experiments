"""在 PyCharm 里运行此文件，重做第 7 节实验并生成图表。"""
from pathlib import Path
import subprocess
import sys


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    steps = [
        ("重算 980 次主实验", ["run.py", "--workers", "4"]),
        ("生成八面板总图、八张单图、四组双栏图和最快方法表", ["present.py"]),
    ]
    for title, args in steps:
        print(title, flush=True)
        subprocess.run([sys.executable, *args], cwd=root, check=True)
    print(f"完成。总图在 {root / 'figures/all_families_grid.pdf'}，八张单图在 {root / 'figures/single'}，最快方法表在 {root / 'tables'}。")
