"""
AutoDL Connector - 连接和管理AutoDL服务器的工具包
"""

__version__ = "1.0.0"
__author__ = "AutoDL Connector Team"

from .connector import AutoDLConnector
from .config_manager import ConfigManager
from .file_transfer import FileTransfer
from .task_manager import TaskManager

__all__ = [
    "AutoDLConnector",
    "ConfigManager",
    "FileTransfer",
    "TaskManager",
]