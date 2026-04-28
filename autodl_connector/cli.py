#!/usr/bin/env python3
"""
AutoDL Connector 命令行接口
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import Optional

from .src import AutoDLConnector, ConfigManager, FileTransfer, TaskManager
from .src.config_manager import ServerConfig, AuthMethod

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AutoDLCli:
    """AutoDL命令行接口"""
    
    def __init__(self):
        self.config_manager = ConfigManager()
    
    def run(self):
        """运行命令行接口"""
        parser = argparse.ArgumentParser(
            description="AutoDL Connector - 连接和管理AutoDL服务器",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
示例:
  %(prog)s connect --server my_server
  %(prog)s upload --server my_server --local ./data --remote /root/data
  %(prog)s exec --server my_server --command "nvidia-smi"
  %(prog)s config add --name my_server --host region-1.autodl.com --username root --password mypass
            """
        )
        
        subparsers = parser.add_subparsers(dest='command', help='命令')
        
        # connect 命令
        connect_parser = subparsers.add_parser('connect', help='连接到服务器')
        connect_parser.add_argument('--server', required=True, help='服务器名称')
        connect_parser.add_argument('--info', action='store_true', help='显示服务器信息')
        
        # upload 命令
        upload_parser = subparsers.add_parser('upload', help='上传文件或目录')
        upload_parser.add_argument('--server', required=True, help='服务器名称')
        upload_parser.add_argument('--local', required=True, help='本地路径')
        upload_parser.add_argument('--remote', required=True, help='远程路径')
        upload_parser.add_argument('--overwrite', action='store_true', help='覆盖已存在的文件')
        
        # download 命令
        download_parser = subparsers.add_parser('download', help='下载文件或目录')
        download_parser.add_argument('--server', required=True, help='服务器名称')
        download_parser.add_argument('--remote', required=True, help='远程路径')
        download_parser.add_argument('--local', required=True, help='本地路径')
        download_parser.add_argument('--overwrite', action='store_true', help='覆盖已存在的文件')
        
        # exec 命令
        exec_parser = subparsers.add_parser('exec', help='执行远程命令')
        exec_parser.add_argument('--server', required=True, help='服务器名称')
        exec_parser.add_argument('--command', required=True, help='要执行的命令')
        exec_parser.add_argument('--timeout', type=int, default=30, help='超时时间（秒）')
        
        # config 命令组
        config_parser = subparsers.add_parser('config', help='配置管理')
        config_subparsers = config_parser.add_subparsers(dest='config_command', help='配置命令')
        
        # config list
        config_list_parser = config_subparsers.add_parser('list', help='列出所有服务器配置')
        
        # config show
        config_show_parser = config_subparsers.add_parser('show', help='显示服务器配置')
        config_show_parser.add_argument('--name', required=True, help='服务器名称')
        
        # config add
        config_add_parser = config_subparsers.add_parser('add', help='添加服务器配置')
        config_add_parser.add_argument('--name', required=True, help='服务器名称')
        config_add_parser.add_argument('--host', required=True, help='服务器地址')
        config_add_parser.add_argument('--port', type=int, default=22, help='SSH端口')
        config_add_parser.add_argument('--username', default='root', help='用户名')
        config_add_parser.add_argument('--auth', choices=['password', 'ssh_key'], default='password', help='认证方式')
        config_add_parser.add_argument('--password', help='密码（用于密码认证）')
        config_add_parser.add_argument('--key-path', help='SSH密钥路径（用于密钥认证）')
        config_add_parser.add_argument('--description', help='描述')
        config_add_parser.add_argument('--tags', help='标签，用逗号分隔')
        
        # config remove
        config_remove_parser = config_subparsers.add_parser('remove', help='移除服务器配置')
        config_remove_parser.add_argument('--name', required=True, help='服务器名称')
        
        # config export
        config_export_parser = config_subparsers.add_parser('export', help='导出配置')
        config_export_parser.add_argument('--file', required=True, help='导出文件路径')
        config_export_parser.add_argument('--format', choices=['json', 'yaml'], default='json', help='导出格式')
        
        # config import
        config_import_parser = config_subparsers.add_parser('import', help='导入配置')
        config_import_parser.add_argument('--file', required=True, help='导入文件路径')
        config_import_parser.add_argument('--format', choices=['json', 'yaml'], default='json', help='导入格式')
        config_import_parser.add_argument('--merge', action='store_true', help='合并现有配置')
        
        # task 命令组
        task_parser = subparsers.add_parser('task', help='任务管理')
        task_subparsers = task_parser.add_subparsers(dest='task_command', help='任务命令')
        
        # task submit
        task_submit_parser = task_subparsers.add_parser('submit', help='提交任务')
        task_submit_parser.add_argument('--server', required=True, help='服务器名称')
        task_submit_parser.add_argument('--command', required=True, help='要执行的命令')
        task_submit_parser.add_argument('--timeout', type=int, default=3600, help='超时时间（秒）')
        task_submit_parser.add_argument('--working-dir', help='工作目录')
        
        # task status
        task_status_parser = task_subparsers.add_parser('status', help='查看任务状态')
        task_status_parser.add_argument('--task-id', help='任务ID（如果不提供则显示所有任务）')
        
        # task cancel
        task_cancel_parser = task_subparsers.add_parser('cancel', help='取消任务')
        task_cancel_parser.add_argument('--task-id', required=True, help='任务ID')
        
        # 解析参数
        args = parser.parse_args()
        
        if not args.command:
            parser.print_help()
            sys.exit(1)
        
        try:
            # 执行命令
            if args.command == 'connect':
                self.handle_connect(args)
            elif args.command == 'upload':
                self.handle_upload(args)
            elif args.command == 'download':
                self.handle_download(args)
            elif args.command == 'exec':
                self.handle_exec(args)
            elif args.command == 'config':
                self.handle_config(args)
            elif args.command == 'task':
                self.handle_task(args)
            else:
                parser.print_help()
                
        except Exception as e:
            logger.error(f"命令执行失败: {e}")
            sys.exit(1)
    
    def _get_connector(self, server_name: str) -> AutoDLConnector:
        """获取连接器实例"""
        server_config = self.config_manager.get_server(server_name)
        if not server_config:
            raise ValueError(f"服务器配置不存在: {server_name}")
        
        # 创建连接器
        connector = AutoDLConnector(
            host=server_config.host,
            port=server_config.port,
            username=server_config.username,
            password=server_config.password if server_config.auth_method == AuthMethod.PASSWORD else None,
            key_path=server_config.key_path if server_config.auth_method == AuthMethod.SSH_KEY else None
        )
        
        return connector
    
    def handle_connect(self, args):
        """处理connect命令"""
        connector = self._get_connector(args.server)
        
        try:
            # 连接服务器
            if connector.connect():
                print(f"✓ 成功连接到服务器: {args.server}")
                
                if args.info:
                    # 显示服务器信息
                    info = connector.get_server_info()
                    print("\n服务器信息:")
                    print("-" * 50)
                    for key, value in info.items():
                        print(f"{key:15}: {value}")
                    print("-" * 50)
            else:
                print(f"✗ 连接服务器失败: {args.server}")
                sys.exit(1)
                
        finally:
            connector.disconnect()
    
    def handle_upload(self, args):
        """处理upload命令"""
        connector = self._get_connector(args.server)
        file_transfer = FileTransfer(connector)
        
        try:
            if connector.connect():
                local_path = Path(args.local)
                if local_path.is_dir():
                    # 上传目录
                    print(f"正在上传目录: {args.local} -> {args.remote}")
                    success, total = file_transfer.upload_directory(
                        args.local, args.remote, args.overwrite
                    )
                    print(f"✓ 目录上传完成: {success}/{total} 个文件成功")
                else:
                    # 上传文件
                    print(f"正在上传文件: {args.local} -> {args.remote}")
                    if file_transfer.upload_file(args.local, args.remote, args.overwrite):
                        print("✓ 文件上传成功")
                    else:
                        print("✗ 文件上传失败")
                        sys.exit(1)
            else:
                print(f"✗ 连接服务器失败: {args.server}")
                sys.exit(1)
                
        finally:
            connector.disconnect()
    
    def handle_download(self, args):
        """处理download命令"""
        connector = self._get_connector(args.server)
        file_transfer = FileTransfer(connector)
        
        try:
            if connector.connect():
                # 下载文件或目录
                print(f"正在下载: {args.remote} -> {args.local}")
                
                # 检查远程路径是文件还是目录
                try:
                    remote_attrs = connector.sftp.stat(args.remote)
                    if remote_attrs.st_mode & 0o40000:  # 目录
                        success, total = file_transfer.download_directory(
                            args.remote, args.local, args.overwrite
                        )
                        print(f"✓ 目录下载完成: {success}/{total} 个文件成功")
                    else:  # 文件
                        if file_transfer.download_file(args.remote, args.local, args.overwrite):
                            print("✓ 文件下载成功")
                        else:
                            print("✗ 文件下载失败")
                            sys.exit(1)
                except FileNotFoundError:
                    print(f"✗ 远程路径不存在: {args.remote}")
                    sys.exit(1)
            else:
                print(f"✗ 连接服务器失败: {args.server}")
                sys.exit(1)
                
        finally:
            connector.disconnect()
    
    def handle_exec(self, args):
        """处理exec命令"""
        connector = self._get_connector(args.server)
        
        try:
            if connector.connect():
                print(f"执行命令: {args.command}")
                result = connector.execute_command(args.command, args.timeout)
                
                if result['success']:
                    print("✓ 命令执行成功")
                    if result['output']:
                        print("\n输出:")
                        print("-" * 50)
                        print(result['output'])
                        print("-" * 50)
                else:
                    print("✗ 命令执行失败")
                    if result['error']:
                        print(f"错误: {result['error']}")
                    sys.exit(1)
            else:
                print(f"✗ 连接服务器失败: {args.server}")
                sys.exit(1)
                
        finally:
            connector.disconnect()
    
    def handle_config(self, args):
        """处理config命令"""
        if args.config_command == 'list':
            servers = self.config_manager.list_servers()
            if not servers:
                print("没有配置任何服务器")
                return
            
            print("服务器配置列表:")
            print("-" * 80)
            print(f"{'名称':20} {'地址':25} {'用户名':10} {'认证方式':10} {'描述':15}")
            print("-" * 80)
            
            for server in servers:
                auth = server.auth_method.value
                desc = server.description or ""
                print(f"{server.name:20} {server.host:25} {server.username:10} {auth:10} {desc:15}")
            print("-" * 80)
            
        elif args.config_command == 'show':
            server = self.config_manager.get_server(args.name)
            if not server:
                print(f"服务器配置不存在: {args.name}")
                sys.exit(1)
            
            print(f"服务器配置: {args.name}")
            print("-" * 50)
            print(f"名称:       {server.name}")
            print(f"地址:       {server.host}:{server.port}")
            print(f"用户名:     {server.username}")
            print(f"认证方式:   {server.auth_method.value}")
            
            if server.auth_method == AuthMethod.PASSWORD:
                print(f"密码:       {'*' * len(server.password) if server.password else '未设置'}")
            else:
                print(f"密钥路径:   {server.key_path}")
            
            print(f"描述:       {server.description or '无'}")
            print(f"标签:       {', '.join(server.tags) if server.tags else '无'}")
            print("-" * 50)
            
        elif args.config_command == 'add':
            # 创建服务器配置
            tags = args.tags.split(',') if args.tags else []
            
            server = ServerConfig(
                name=args.name,
                host=args.host,
                port=args.port,
                username=args.username,
                auth_method=AuthMethod(args.auth),
                password=args.password if args.auth == 'password' else None,
                key_path=args.key_path if args.auth == 'ssh_key' else None,
                description=args.description,
                tags=tags
            )
            
            # 验证配置
            errors = self.config_manager.validate_server_config(server)
            if errors:
                print("配置验证失败:")
                for error in errors:
                    print(f"  - {error}")
                sys.exit(1)
            
            # 添加配置
            if self.config_manager.add_server(server):
                print(f"✓ 服务器配置已添加: {args.name}")
            else:
                print(f"✗ 添加服务器配置失败: {args.name}")
                sys.exit(1)
            
        elif args.config_command == 'remove':
            if self.config_manager.remove_server(args.name):
                print(f"✓ 服务器配置已移除: {args.name}")
            else:
                print(f"✗ 移除服务器配置失败: {args.name}")
                sys.exit(1)
            
        elif args.config_command == 'export':
            if self.config_manager.export_config(args.file, args.format):
                print(f"✓ 配置已导出到: {args.file}")
            else:
                print(f"✗ 导出配置失败")
                sys.exit(1)
            
        elif args.config_command == 'import':
            if self.config_manager.import_config(args.file, args.format, args.merge):
                print(f"✓ 配置已从文件导入: {args.file}")
            else:
                print(f"✗ 导入配置失败")
                sys.exit(1)
    
    def handle_task(self, args):
        """处理task命令"""
        # 这里需要实现任务管理功能
        # 由于任务管理需要持久化存储，这里只提供基本功能
        
        if args.task_command == 'submit':
            print("任务提交功能需要完整的任务管理器实现")
            print("请使用Python API进行任务管理")
            
        elif args.task_command == 'status':
            print("任务状态功能需要完整的任务管理器实现")
            print("请使用Python API进行任务管理")
            
        elif args.task_command == 'cancel':
            print("任务取消功能需要完整的任务管理器实现")
            print("请使用Python API进行任务管理")


def main():
    """主函数"""
    cli = AutoDLCli()
    cli.run()


if __name__ == '__main__':
    main()