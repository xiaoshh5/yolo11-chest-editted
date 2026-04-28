"""
AutoDL连接器 - 提供SSH连接和远程命令执行功能
"""

import paramiko
import socket
import time
from typing import Optional, Dict, Any, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class AutoDLConnector:
    """AutoDL服务器连接器"""
    
    def __init__(self, host: str, port: int = 22, username: str = "root", 
                 password: Optional[str] = None, key_path: Optional[str] = None):
        """
        初始化AutoDL连接器
        
        Args:
            host: 服务器地址
            port: SSH端口，默认为22
            username: 用户名，默认为root
            password: 密码（可选）
            key_path: SSH密钥路径（可选）
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.key_path = key_path
        self.client: Optional[paramiko.SSHClient] = None
        self.sftp: Optional[paramiko.SFTPClient] = None
        self.is_connected = False
        
    def connect(self, timeout: int = 10) -> bool:
        """
        连接到AutoDL服务器
        
        Args:
            timeout: 连接超时时间（秒）
            
        Returns:
            bool: 连接是否成功
        """
        try:
            logger.info(f"正在连接到 {self.username}@{self.host}:{self.port}")
            
            self.client = paramiko.SSHClient()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # 尝试使用密钥或密码连接
            if self.key_path and Path(self.key_path).exists():
                logger.info(f"使用密钥文件: {self.key_path}")
                key = paramiko.RSAKey.from_private_key_file(self.key_path)
                self.client.connect(
                    hostname=self.host,
                    port=self.port,
                    username=self.username,
                    pkey=key,
                    timeout=timeout,
                    banner_timeout=timeout
                )
            elif self.password:
                logger.info("使用密码连接")
                self.client.connect(
                    hostname=self.host,
                    port=self.port,
                    username=self.username,
                    password=self.password,
                    timeout=timeout,
                    banner_timeout=timeout
                )
            else:
                raise ValueError("必须提供密码或密钥文件路径")
            
            # 创建SFTP客户端
            self.sftp = self.client.open_sftp()
            self.is_connected = True
            
            logger.info(f"成功连接到 {self.host}")
            return True
            
        except (paramiko.AuthenticationException, paramiko.SSHException, 
                socket.timeout, socket.error) as e:
            logger.error(f"连接失败: {e}")
            self.disconnect()
            return False
    
    def disconnect(self):
        """断开连接"""
        if self.sftp:
            try:
                self.sftp.close()
            except:
                pass
            self.sftp = None
            
        if self.client:
            try:
                self.client.close()
            except:
                pass
            self.client = None
            
        self.is_connected = False
        logger.info("连接已关闭")
    
    def execute_command(self, command: str, timeout: int = 30) -> Dict[str, Any]:
        """
        执行远程命令
        
        Args:
            command: 要执行的命令
            timeout: 命令执行超时时间（秒）
            
        Returns:
            Dict: 包含输出、错误和返回码的字典
        """
        if not self.is_connected or not self.client:
            raise ConnectionError("未连接到服务器")
        
        try:
            logger.debug(f"执行命令: {command}")
            
            stdin, stdout, stderr = self.client.exec_command(command, timeout=timeout)
            
            # 读取输出
            output = stdout.read().decode('utf-8', errors='ignore').strip()
            error = stderr.read().decode('utf-8', errors='ignore').strip()
            exit_code = stdout.channel.recv_exit_status()
            
            result = {
                'command': command,
                'output': output,
                'error': error,
                'exit_code': exit_code,
                'success': exit_code == 0
            }
            
            if exit_code != 0:
                logger.warning(f"命令执行失败 (exit_code={exit_code}): {error}")
            else:
                logger.debug(f"命令执行成功")
                
            return result
            
        except paramiko.SSHException as e:
            logger.error(f"命令执行异常: {e}")
            return {
                'command': command,
                'output': '',
                'error': str(e),
                'exit_code': -1,
                'success': False
            }
    
    def check_connection(self) -> bool:
        """
        检查连接状态
        
        Returns:
            bool: 连接是否正常
        """
        if not self.is_connected or not self.client:
            return False
        
        try:
            # 执行一个简单的命令来测试连接
            result = self.execute_command("echo 'test'", timeout=5)
            return result['success']
        except:
            return False
    
    def get_server_info(self) -> Dict[str, Any]:
        """
        获取服务器信息
        
        Returns:
            Dict: 服务器信息
        """
        info = {}
        
        # 获取系统信息
        commands = {
            'hostname': 'hostname',
            'os': 'cat /etc/os-release | grep PRETTY_NAME',
            'kernel': 'uname -r',
            'cpu': 'lscpu | grep "Model name"',
            'memory': 'free -h | grep Mem',
            'disk': 'df -h /',
            'gpu': 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU"',
            'python': 'python3 --version 2>/dev/null || python --version 2>/dev/null || echo "Python not found"'
        }
        
        for key, cmd in commands.items():
            try:
                result = self.execute_command(cmd, timeout=10)
                if result['success']:
                    info[key] = result['output'].strip()
                else:
                    info[key] = f"Error: {result['error']}"
            except Exception as e:
                info[key] = f"Exception: {str(e)}"
        
        return info
    
    def __enter__(self):
        """上下文管理器入口"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.disconnect()
    
    def __del__(self):
        """析构函数"""
        self.disconnect()