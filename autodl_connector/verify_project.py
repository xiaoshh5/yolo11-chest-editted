#!/usr/bin/env python3
"""
验证AutoDL Connector项目结构
"""

import os
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """检查文件是否存在"""
    if Path(filepath).exists():
        print(f"[OK] {description}: {filepath}")
        return True
    else:
        print(f"[FAIL] {description}: {filepath} (不存在)")
        return False

def check_directory_exists(dirpath, description):
    """检查目录是否存在"""
    if Path(dirpath).exists() and Path(dirpath).is_dir():
        print(f"[OK] {description}: {dirpath}")
        return True
    else:
        print(f"[FAIL] {description}: {dirpath} (不存在)")
        return False

def main():
    """验证项目结构"""
    print("=" * 60)
    print("AutoDL Connector 项目结构验证")
    print("=" * 60)
    
    project_root = Path(__file__).parent
    print(f"项目根目录: {project_root}")
    
    # 检查主要文件
    files_to_check = [
        (project_root / "__init__.py", "包入口文件"),
        (project_root / "cli.py", "命令行接口"),
        (project_root / "README.md", "文档"),
        (project_root / "requirements.txt", "依赖文件"),
        (project_root / "setup.py", "安装脚本"),
        (project_root / "test_basic.py", "测试脚本"),
    ]
    
    files_passed = 0
    for filepath, description in files_to_check:
        if check_file_exists(filepath, description):
            files_passed += 1
    
    # 检查目录
    dirs_to_check = [
        (project_root / "src", "源代码目录"),
        (project_root / "examples", "示例目录"),
        (project_root / "configs", "配置目录"),
        (project_root / "tests", "测试目录"),
    ]
    
    dirs_passed = 0
    for dirpath, description in dirs_to_check:
        if check_directory_exists(dirpath, description):
            dirs_passed += 1
    
    # 检查src目录下的文件
    src_files = [
        (project_root / "src" / "__init__.py", "src包入口"),
        (project_root / "src" / "connector.py", "连接器模块"),
        (project_root / "src" / "file_transfer.py", "文件传输模块"),
        (project_root / "src" / "task_manager.py", "任务管理器模块"),
        (project_root / "src" / "config_manager.py", "配置管理器模块"),
    ]
    
    src_files_passed = 0
    for filepath, description in src_files:
        if check_file_exists(filepath, description):
            src_files_passed += 1
    
    # 检查examples目录
    example_files = [
        (project_root / "examples" / "basic_usage.py", "基本使用示例"),
    ]
    
    example_files_passed = 0
    for filepath, description in example_files:
        if check_file_exists(filepath, description):
            example_files_passed += 1
    
    print("\n" + "=" * 60)
    print("验证结果:")
    print(f"主要文件: {files_passed}/{len(files_to_check)} 通过")
    print(f"目录结构: {dirs_passed}/{len(dirs_to_check)} 通过")
    print(f"源代码文件: {src_files_passed}/{len(src_files)} 通过")
    print(f"示例文件: {example_files_passed}/{len(example_files)} 通过")
    
    total_checks = len(files_to_check) + len(dirs_to_check) + len(src_files) + len(example_files)
    total_passed = files_passed + dirs_passed + src_files_passed + example_files_passed
    
    print(f"\n总计: {total_passed}/{total_checks} 通过")
    
    if total_passed == total_checks:
        print("\n[SUCCESS] 项目结构完整!")
        print("\n下一步:")
        print("1. 安装依赖: pip install -r requirements.txt")
        print("2. 安装包: pip install -e .")
        print("3. 查看示例: python examples/basic_usage.py")
        print("4. 使用CLI: python -m autodl_connector.cli --help")
        return 0
    else:
        print(f"\n[FAILED] 有 {total_checks - total_passed} 个检查失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())