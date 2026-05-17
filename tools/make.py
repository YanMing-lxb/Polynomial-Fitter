"""
=======================================================================
 ····Y88b···d88P················888b·····d888·d8b·······················
 ·····Y88b·d88P·················8888b···d8888·Y8P······················
 ······Y88o88P··················88888b·d88888··························
 ·······Y888P··8888b···88888b···888Y88888P888·888·88888b·····d88b······
 ········888······"88b·888·"88b·888·Y888P·888·888·888·"88b·d88P"88b·····
 ········888···d888888·888··888·888··Y8P··888·888·888··888·888··888·····
 ········888··888··888·888··888·888···"···888·888·888··888·Y88b·888·····
 ········888··"Y888888·888··888·888·······888·888·888··888··"Y88888·····
 ·······························································888·····
 ··························································Y8b·d88P·····
 ···························································"Y88P"·····
=======================================================================

-----------------------------------------------------------------------
Author       : 焱铭
Date         : 2026-05-17
Github       : https://github.com/YanMing-lxb/
FilePath     : /Polynomial-Fitter/tools/make.py
Description  :
-----------------------------------------------------------------------
"""

import shutil
import subprocess
import sys
from pathlib import Path

from utils import console, run_command

from config import __version__


def inswhl():
    console.print("📦 开始安装测试 Polynomial-Fitter", style="status")

    uninstall_success = run_command(
        command=["uv", "pip", "uninstall", "polynomial-fitter"],
        success_msg="旧版 Polynomial-Fitter 卸载完成",
        error_msg="旧版 Polynomial-Fitter 卸载失败",
        process_name="卸载旧版 Polynomial-Fitter",
    )

    whl_files = list(Path("dist").glob("*.whl"))
    if not whl_files:
        raise FileNotFoundError("dist 目录中没有找到 .whl 文件")
    install_success = run_command(
        command=["uv", "pip", "install", str(whl_files[0])],
        success_msg="测试 Polynomial-Fitter 安装完成",
        error_msg="测试 Polynomial-Fitter 安装失败",
        process_name="安装测试版 Polynomial-Fitter",
    )
    return uninstall_success and install_success


def upload():
    tag_name = f"v{__version__}"

    run_command(
        command=["git", "tag", tag_name],
        success_msg=f"标签 {tag_name} 创建成功",
        error_msg=f"标签 {tag_name} 创建失败",
        process_name="创建标签",
    )
    console.log(f"创建标签: {tag_name}")

    run_command(
        command=["git", "push", "origin", tag_name],
        success_msg=f"标签 {tag_name} 推送成功",
        error_msg=f"标签 {tag_name} 推送失败",
        process_name="推送标签",
    )
    console.log(f"推送标签: {tag_name}")

    console.log("成功上传标签和推送到远程仓库，发布到 github")


def html():
    readme_md = Path("README.md")
    readme_html = Path("README.html")
    target_dir = Path("src/assets")

    run_command(
        command=["pandoc", str(readme_md), "-o", str(readme_html)],
        success_msg="README.html 生成成功",
        error_msg="README.html 生成失败",
        process_name="生成 README.html",
    )

    if not target_dir.exists():
        target_dir.mkdir(parents=True)
        console.log(f"已创建目录: {target_dir}")

    target_html = target_dir / "README.html"
    shutil.move(str(readme_html), str(target_html))
    console.log(f"生成 HTML 并移动到 {target_html}")


def main():
    targets = {
        "upload": upload,
        "inswhl": inswhl,
        "html": html,
    }

    if len(sys.argv) < 2 or sys.argv[1] not in targets:
        console.log(f"用法: {sys.argv[0]} <目标>")
        console.log("可用目标:", ", ".join(targets.keys()))
        sys.exit(1)

    target = sys.argv[1]
    try:
        targets[target]()
    except subprocess.CalledProcessError as e:
        console.log(f"执行命令时出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
