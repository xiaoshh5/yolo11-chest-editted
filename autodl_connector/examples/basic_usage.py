"""
AutoDL Connector 基本使用示例
"""

import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO)

# 导入AutoDL Connector
from autodl_connector import AutoDLConnector, ConfigManager, FileTransfer, TaskManager
from autodl_connector.src.config_manager import ServerConfig, AuthMethod


def example_connect():
    """示例1: 连接到服务器"""
    print("=" * 60)
    print("示例1: 连接到服务器")
    print("=" * 60)
    
    # 创建连接器
    connector = AutoDLConnector(
        host="region-1.autodl.com",  # 替换为实际的服务器地址
        port=22,
        username="root",
        password="your_password"  # 替换为实际的密码
    )
    
    try:
        # 连接服务器
        if connector.connect():
            print("✓ 连接成功")
            
            # 获取服务器信息
            info = connector.get_server_info()
            print("\n服务器信息:")
            for key, value in info.items():
                print(f"  {key}: {value}")
            
            # 执行命令
            result = connector.execute_command("nvidia-smi")
            if result['success']:
                print(f"\nGPU信息:\n{result['output']}")
            else:
                print(f"\n获取GPU信息失败: {result['error']}")
        else:
            print("✗ 连接失败")
            
    finally:
        connector.disconnect()
        print("\n连接已关闭")


def example_config_manager():
    """示例2: 配置管理器"""
    print("\n" + "=" * 60)
    print("示例2: 配置管理器")
    print("=" * 60)
    
    # 创建配置管理器
    config_manager = ConfigManager()
    
    # 添加服务器配置
    server = ServerConfig(
        name="my_server",
        host="region-1.autodl.com",
        port=22,
        username="root",
        auth_method=AuthMethod.PASSWORD,
        password="your_password",
        description="我的AutoDL服务器",
        tags=["gpu", "training"]
    )
    
    if config_manager.add_server(server):
        print("✓ 服务器配置已添加")
    
    # 列出所有服务器
    servers = config_manager.list_servers()
    print(f"\n已配置的服务器 ({len(servers)} 个):")
    for s in servers:
        print(f"  - {s.name}: {s.host} ({s.description})")
    
    # 导出配置
    config_manager.export_config("autodl_config.json")
    print("\n✓ 配置已导出到 autodl_config.json")


def example_file_transfer():
    """示例3: 文件传输"""
    print("\n" + "=" * 60)
    print("示例3: 文件传输")
    print("=" * 60)
    
    # 创建连接器
    connector = AutoDLConnector(
        host="region-1.autodl.com",
        username="root",
        password="your_password"
    )
    
    try:
        if connector.connect():
            # 创建文件传输管理器
            file_transfer = FileTransfer(connector)
            
            # 创建测试文件
            test_file = Path("test_upload.txt")
            test_file.write_text("这是一个测试文件，用于演示文件上传功能。")
            
            # 上传文件
            if file_transfer.upload_file("test_upload.txt", "/root/test_upload.txt"):
                print("✓ 文件上传成功")
            
            # 下载文件
            if file_transfer.download_file("/root/test_upload.txt", "test_download.txt"):
                print("✓ 文件下载成功")
                
                # 验证文件内容
                downloaded_content = Path("test_download.txt").read_text()
                print(f"下载的文件内容: {downloaded_content}")
            
            # 清理测试文件
            test_file.unlink(missing_ok=True)
            Path("test_download.txt").unlink(missing_ok=True)
            
    finally:
        connector.disconnect()


def example_task_manager():
    """示例4: 任务管理"""
    print("\n" + "=" * 60)
    print("示例4: 任务管理")
    print("=" * 60)
    
    # 创建连接器
    connector = AutoDLConnector(
        host="region-1.autodl.com",
        username="root",
        password="your_password"
    )
    
    try:
        if connector.connect():
            # 创建任务管理器
            task_manager = TaskManager(connector)
            
            # 提交任务
            task_id = task_manager.submit_task(
                command="python -c 'import time; time.sleep(2); print(\"任务完成\")'",
                task_id="test_task",
                timeout=10
            )
            
            print(f"✓ 任务已提交: {task_id}")
            
            # 等待任务完成
            task = task_manager.wait_for_task(task_id, timeout=5)
            if task:
                print(f"任务状态: {task.status.value}")
                if task.result:
                    print(f"任务输出: {task.result['output']}")
            
            # 启动任务监控
            task_manager.monitor_tasks()
            print("✓ 任务监控已启动")
            
    finally:
        connector.disconnect()


def example_context_manager():
    """示例5: 使用上下文管理器"""
    print("\n" + "=" * 60)
    print("示例5: 使用上下文管理器")
    print("=" * 60)
    
    # 使用上下文管理器自动管理连接
    with AutoDLConnector(
        host="region-1.autodl.com",
        username="root",
        password="your_password"
    ) as connector:
        
        if connector.is_connected:
            print("✓ 使用上下文管理器连接成功")
            
            # 执行多个命令
            commands = [
                "hostname",
                "whoami",
                "date",
                "free -h"
            ]
            
            for cmd in commands:
                result = connector.execute_command(cmd)
                print(f"\n命令: {cmd}")
                print(f"输出: {result['output']}")
    
    print("\n连接已自动关闭")


def main():
    """运行所有示例"""
    print("AutoDL Connector 使用示例")
    print("=" * 60)
    
    try:
        # 注意：以下示例需要实际的AutoDL服务器信息才能运行
        # 请先配置好服务器信息
        
        # example_connect()
        # example_config_manager()
        # example_file_transfer()
        # example_task_manager()
        # example_context_manager()
        
        print("\n" + "=" * 60)
        print("示例代码已准备就绪")
        print("=" * 60)
        print("\n要运行示例，请:")
        print("1. 取消注释示例函数调用")
        print("2. 替换服务器地址、用户名和密码")
        print("3. 运行: python basic_usage.py")
        
    except Exception as e:
        print(f"\n示例执行出错: {e}")
        print("请确保已安装所有依赖包: pip install paramiko tqdm pyyaml")


if __name__ == "__main__":
    main()