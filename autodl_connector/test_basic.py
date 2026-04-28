#!/usr/bin/env python3
"""
AutoDL Connector 基本功能测试
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试模块导入"""
    print("测试模块导入...")
    
    try:
        from autodl_connector import AutoDLConnector, ConfigManager, FileTransfer, TaskManager
        from autodl_connector.src.config_manager import ServerConfig, AuthMethod
        
        print("✓ 模块导入成功")
        return True
    except ImportError as e:
        print(f"✗ 模块导入失败: {e}")
        return False

def test_config_manager():
    """测试配置管理器"""
    print("\n测试配置管理器...")
    
    try:
        from autodl_connector import ConfigManager
        from autodl_connector.src.config_manager import ServerConfig, AuthMethod
        
        # 创建临时配置目录
        import tempfile
        temp_dir = tempfile.mkdtemp()
        
        config_manager = ConfigManager(config_dir=temp_dir)
        
        # 创建服务器配置
        server = ServerConfig(
            name="test_server",
            host="test.autodl.com",
            port=22,
            username="testuser",
            auth_method=AuthMethod.PASSWORD,
            password="testpass",
            description="测试服务器"
        )
        
        # 添加服务器配置
        if config_manager.add_server(server):
            print("✓ 添加服务器配置成功")
        else:
            print("✗ 添加服务器配置失败")
            return False
        
        # 获取服务器配置
        retrieved_server = config_manager.get_server("test_server")
        if retrieved_server and retrieved_server.name == "test_server":
            print("✓ 获取服务器配置成功")
        else:
            print("✗ 获取服务器配置失败")
            return False
        
        # 列出服务器
        servers = config_manager.list_servers()
        if len(servers) == 1:
            print("✓ 列出服务器成功")
        else:
            print("✗ 列出服务器失败")
            return False
        
        # 清理临时目录
        import shutil
        shutil.rmtree(temp_dir)
        
        return True
        
    except Exception as e:
        print(f"✗ 配置管理器测试失败: {e}")
        return False

def test_connector_structure():
    """测试连接器结构"""
    print("\n测试连接器结构...")
    
    try:
        from autodl_connector import AutoDLConnector
        
        # 创建连接器实例
        connector = AutoDLConnector(
            host="test.autodl.com",
            username="testuser",
            password="testpass"
        )
        
        # 检查属性
        required_attrs = ['host', 'port', 'username', 'password', 'client', 'sftp', 'is_connected']
        for attr in required_attrs:
            if hasattr(connector, attr):
                print(f"✓ 属性 {attr} 存在")
            else:
                print(f"✗ 属性 {attr} 不存在")
                return False
        
        # 检查方法
        required_methods = ['connect', 'disconnect', 'execute_command', 'check_connection', 'get_server_info']
        for method in required_methods:
            if hasattr(connector, method) and callable(getattr(connector, method)):
                print(f"✓ 方法 {method} 存在且可调用")
            else:
                print(f"✗ 方法 {method} 不存在或不可调用")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ 连接器结构测试失败: {e}")
        return False

def test_file_transfer_structure():
    """测试文件传输结构"""
    print("\n测试文件传输结构...")
    
    try:
        from autodl_connector import AutoDLConnector, FileTransfer
        
        # 创建连接器实例
        connector = AutoDLConnector(
            host="test.autodl.com",
            username="testuser",
            password="testpass"
        )
        
        # 创建文件传输实例
        file_transfer = FileTransfer(connector)
        
        # 检查方法
        required_methods = ['upload_file', 'download_file', 'upload_directory', 'download_directory', 'compare_files']
        for method in required_methods:
            if hasattr(file_transfer, method) and callable(getattr(file_transfer, method)):
                print(f"✓ 方法 {method} 存在且可调用")
            else:
                print(f"✗ 方法 {method} 不存在或不可调用")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ 文件传输结构测试失败: {e}")
        return False

def test_task_manager_structure():
    """测试任务管理器结构"""
    print("\n测试任务管理器结构...")
    
    try:
        from autodl_connector import AutoDLConnector, TaskManager
        
        # 创建连接器实例
        connector = AutoDLConnector(
            host="test.autodl.com",
            username="testuser",
            password="testpass"
        )
        
        # 创建任务管理器实例
        task_manager = TaskManager(connector)
        
        # 检查方法
        required_methods = ['submit_task', 'get_task', 'cancel_task', 'wait_for_task', 'monitor_tasks']
        for method in required_methods:
            if hasattr(task_manager, method) and callable(getattr(task_manager, method)):
                print(f"✓ 方法 {method} 存在且可调用")
            else:
                print(f"✗ 方法 {method} 不存在或不可调用")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ 任务管理器结构测试失败: {e}")
        return False

def test_cli_import():
    """测试CLI导入"""
    print("\n测试CLI导入...")
    
    try:
        # 检查CLI文件是否存在
        cli_file = Path(__file__).parent / "cli.py"
        if cli_file.exists():
            print("✓ CLI文件存在")
        else:
            print("✗ CLI文件不存在")
            return False
        
        # 尝试导入CLI模块
        import importlib.util
        spec = importlib.util.spec_from_file_location("cli", str(cli_file))
        cli_module = importlib.util.module_from_spec(spec)
        
        try:
            spec.loader.exec_module(cli_module)
            print("✓ CLI模块导入成功")
            
            # 检查必要的类
            if hasattr(cli_module, 'AutoDLCli'):
                print("✓ AutoDLCli类存在")
            else:
                print("✗ AutoDLCli类不存在")
                return False
            
            if hasattr(cli_module, 'main'):
                print("✓ main函数存在")
            else:
                print("✗ main函数不存在")
                return False
            
            return True
            
        except Exception as e:
            print(f"✗ CLI模块执行失败: {e}")
            return False
            
    except Exception as e:
        print(f"✗ CLI导入测试失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("=" * 60)
    print("AutoDL Connector 基本功能测试")
    print("=" * 60)
    
    tests = [
        ("模块导入", test_imports),
        ("配置管理器", test_config_manager),
        ("连接器结构", test_connector_structure),
        ("文件传输结构", test_file_transfer_structure),
        ("任务管理器结构", test_task_manager_structure),
        ("CLI导入", test_cli_import),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {test_name} 测试异常: {e}")
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed}/{total} 通过")
    print("=" * 60)
    
    if passed == total:
        print("\n✓ 所有基本功能测试通过!")
        print("\n项目结构完整，可以开始使用。")
        print("\n下一步:")
        print("1. 安装依赖: pip install -r requirements.txt")
        print("2. 查看示例: python examples/basic_usage.py")
        print("3. 使用CLI: python -m autodl_connector.cli --help")
        return 0
    else:
        print(f"\n✗ 有 {total - passed} 个测试失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())