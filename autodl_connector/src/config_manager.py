"""
配置管理器 - 管理AutoDL连接配置
"""

import json
import yaml
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
from enum import Enum

logger = logging.getLogger(__name__)


class AuthMethod(Enum):
    """认证方法枚举"""
    PASSWORD = "password"
    SSH_KEY = "ssh_key"


@dataclass
class ServerConfig:
    """服务器配置"""
    name: str
    host: str
    port: int = 22
    username: str = "root"
    auth_method: AuthMethod = AuthMethod.PASSWORD
    password: Optional[str] = None
    key_path: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['auth_method'] = self.auth_method.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ServerConfig':
        """从字典创建"""
        data = data.copy()
        data['auth_method'] = AuthMethod(data.get('auth_method', 'password'))
        return cls(**data)


@dataclass
class TransferConfig:
    """传输配置"""
    default_local_dir: str = "./data"
    default_remote_dir: str = "/root/data"
    chunk_size: int = 8192
    max_retries: int = 3
    retry_delay: int = 5
    verify_hash: bool = True
    hash_algorithm: str = "md5"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TransferConfig':
        """从字典创建"""
        return cls(**data)


@dataclass
class TaskConfig:
    """任务配置"""
    default_timeout: int = 3600
    max_concurrent_tasks: int = 5
    task_history_size: int = 100
    auto_save_interval: int = 300  # 秒
    log_level: str = "INFO"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskConfig':
        """从字典创建"""
        return cls(**data)


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_dir: 配置目录路径，如果为None则使用默认目录
        """
        if config_dir is None:
            config_dir = os.path.join(os.path.expanduser("~"), ".autodl_connector")
        
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # 配置文件路径
        self.servers_file = self.config_dir / "servers.json"
        self.transfer_config_file = self.config_dir / "transfer_config.json"
        self.task_config_file = self.config_dir / "task_config.json"
        
        # 加载配置
        self.servers: Dict[str, ServerConfig] = self._load_servers()
        self.transfer_config = self._load_transfer_config()
        self.task_config = self._load_task_config()
    
    def _load_servers(self) -> Dict[str, ServerConfig]:
        """加载服务器配置"""
        if not self.servers_file.exists():
            return {}
        
        try:
            with open(self.servers_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            servers = {}
            for server_data in data.get('servers', []):
                try:
                    server = ServerConfig.from_dict(server_data)
                    servers[server.name] = server
                except Exception as e:
                    logger.error(f"加载服务器配置失败 {server_data.get('name', 'unknown')}: {e}")
            
            logger.info(f"已加载 {len(servers)} 个服务器配置")
            return servers
            
        except Exception as e:
            logger.error(f"加载服务器配置文件失败: {e}")
            return {}
    
    def _load_transfer_config(self) -> TransferConfig:
        """加载传输配置"""
        if not self.transfer_config_file.exists():
            return TransferConfig()
        
        try:
            with open(self.transfer_config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return TransferConfig.from_dict(data)
        except Exception as e:
            logger.error(f"加载传输配置文件失败: {e}")
            return TransferConfig()
    
    def _load_task_config(self) -> TaskConfig:
        """加载任务配置"""
        if not self.task_config_file.exists():
            return TaskConfig()
        
        try:
            with open(self.task_config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return TaskConfig.from_dict(data)
        except Exception as e:
            logger.error(f"加载任务配置文件失败: {e}")
            return TaskConfig()
    
    def save_servers(self):
        """保存服务器配置"""
        try:
            data = {
                'servers': [server.to_dict() for server in self.servers.values()]
            }
            
            with open(self.servers_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"服务器配置已保存到: {self.servers_file}")
            return True
            
        except Exception as e:
            logger.error(f"保存服务器配置失败: {e}")
            return False
    
    def save_transfer_config(self):
        """保存传输配置"""
        try:
            with open(self.transfer_config_file, 'w', encoding='utf-8') as f:
                json.dump(self.transfer_config.to_dict(), f, indent=2, ensure_ascii=False)
            
            logger.info(f"传输配置已保存到: {self.transfer_config_file}")
            return True
            
        except Exception as e:
            logger.error(f"保存传输配置失败: {e}")
            return False
    
    def save_task_config(self):
        """保存任务配置"""
        try:
            with open(self.task_config_file, 'w', encoding='utf-8') as f:
                json.dump(self.task_config.to_dict(), f, indent=2, ensure_ascii=False)
            
            logger.info(f"任务配置已保存到: {self.task_config_file}")
            return True
            
        except Exception as e:
            logger.error(f"保存任务配置失败: {e}")
            return False
    
    def save_all(self):
        """保存所有配置"""
        self.save_servers()
        self.save_transfer_config()
        self.save_task_config()
        logger.info("所有配置已保存")
    
    def add_server(self, server: ServerConfig) -> bool:
        """
        添加服务器配置
        
        Args:
            server: 服务器配置
            
        Returns:
            bool: 是否成功添加
        """
        if server.name in self.servers:
            logger.warning(f"服务器名称已存在: {server.name}")
            return False
        
        self.servers[server.name] = server
        self.save_servers()
        logger.info(f"服务器配置已添加: {server.name}")
        return True
    
    def update_server(self, name: str, **kwargs) -> bool:
        """
        更新服务器配置
        
        Args:
            name: 服务器名称
            **kwargs: 要更新的字段
            
        Returns:
            bool: 是否成功更新
        """
        if name not in self.servers:
            logger.error(f"服务器不存在: {name}")
            return False
        
        server = self.servers[name]
        
        for key, value in kwargs.items():
            if hasattr(server, key):
                if key == 'auth_method' and isinstance(value, str):
                    value = AuthMethod(value)
                setattr(server, key, value)
            else:
                logger.warning(f"服务器配置没有字段: {key}")
        
        self.save_servers()
        logger.info(f"服务器配置已更新: {name}")
        return True
    
    def remove_server(self, name: str) -> bool:
        """
        移除服务器配置
        
        Args:
            name: 服务器名称
            
        Returns:
            bool: 是否成功移除
        """
        if name not in self.servers:
            logger.error(f"服务器不存在: {name}")
            return False
        
        del self.servers[name]
        self.save_servers()
        logger.info(f"服务器配置已移除: {name}")
        return True
    
    def get_server(self, name: str) -> Optional[ServerConfig]:
        """
        获取服务器配置
        
        Args:
            name: 服务器名称
            
        Returns:
            Optional[ServerConfig]: 服务器配置，如果不存在返回None
        """
        return self.servers.get(name)
    
    def list_servers(self) -> List[ServerConfig]:
        """
        列出所有服务器配置
        
        Returns:
            List[ServerConfig]: 服务器配置列表
        """
        return list(self.servers.values())
    
    def search_servers(self, keyword: str) -> List[ServerConfig]:
        """
        搜索服务器配置
        
        Args:
            keyword: 搜索关键词
            
        Returns:
            List[ServerConfig]: 匹配的服务器配置列表
        """
        results = []
        keyword_lower = keyword.lower()
        
        for server in self.servers.values():
            if (keyword_lower in server.name.lower() or
                (server.description and keyword_lower in server.description.lower()) or
                any(keyword_lower in tag.lower() for tag in server.tags)):
                results.append(server)
        
        return results
    
    def export_config(self, filepath: str, format: str = "json") -> bool:
        """
        导出配置到文件
        
        Args:
            filepath: 导出文件路径
            format: 导出格式 (json, yaml)
            
        Returns:
            bool: 是否成功导出
        """
        try:
            config_data = {
                'servers': [server.to_dict() for server in self.servers.values()],
                'transfer_config': self.transfer_config.to_dict(),
                'task_config': self.task_config.to_dict()
            }
            
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == "yaml":
                import yaml
                with open(filepath, 'w', encoding='utf-8') as f:
                    yaml.dump(config_data, f, allow_unicode=True, default_flow_style=False)
            else:  # 默认使用JSON
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"配置已导出到: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"导出配置失败: {e}")
            return False
    
    def import_config(self, filepath: str, format: str = "json", merge: bool = True) -> bool:
        """
        从文件导入配置
        
        Args:
            filepath: 导入文件路径
            format: 导入格式 (json, yaml)
            merge: 是否合并现有配置（True）或替换（False）
            
        Returns:
            bool: 是否成功导入
        """
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                logger.error(f"配置文件不存在: {filepath}")
                return False
            
            if format.lower() == "yaml":
                import yaml
                with open(filepath, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
            else:  # 默认使用JSON
                with open(filepath, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
            
            # 导入服务器配置
            if 'servers' in config_data:
                if not merge:
                    self.servers.clear()
                
                for server_data in config_data['servers']:
                    try:
                        server = ServerConfig.from_dict(server_data)
                        self.servers[server.name] = server
                    except Exception as e:
                        logger.error(f"导入服务器配置失败: {e}")
            
            # 导入传输配置
            if 'transfer_config' in config_data:
                self.transfer_config = TransferConfig.from_dict(config_data['transfer_config'])
            
            # 导入任务配置
            if 'task_config' in config_data:
                self.task_config = TaskConfig.from_dict(config_data['task_config'])
            
            # 保存配置
            self.save_all()
            logger.info(f"配置已从文件导入: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"导入配置失败: {e}")
            return False
    
    def create_default_config(self):
        """创建默认配置"""
        # 添加示例服务器配置
        example_server = ServerConfig(
            name="example",
            host="region-1.autodl.com",
            port=22,
            username="root",
            auth_method=AuthMethod.PASSWORD,
            password="your_password",
            description="示例AutoDL服务器",
            tags=["example", "gpu"]
        )
        
        if "example" not in self.servers:
            self.add_server(example_server)
        
        # 保存所有配置
        self.save_all()
        logger.info("默认配置已创建")
    
    def validate_server_config(self, server: ServerConfig) -> List[str]:
        """
        验证服务器配置
        
        Args:
            server: 服务器配置
            
        Returns:
            List[str]: 错误消息列表，如果没有错误返回空列表
        """
        errors = []
        
        if not server.name.strip():
            errors.append("服务器名称不能为空")
        
        if not server.host.strip():
            errors.append("服务器地址不能为空")
        
        if server.port < 1 or server.port > 65535:
            errors.append("端口号必须在1-65535之间")
        
        if not server.username.strip():
            errors.append("用户名不能为空")
        
        if server.auth_method == AuthMethod.PASSWORD and not server.password:
            errors.append("密码认证需要提供密码")
        
        if server.auth_method == AuthMethod.SSH_KEY:
            if not server.key_path:
                errors.append("SSH密钥认证需要提供密钥文件路径")
            elif not Path(server.key_path).exists():
                errors.append(f"密钥文件不存在: {server.key_path}")
        
        return errors