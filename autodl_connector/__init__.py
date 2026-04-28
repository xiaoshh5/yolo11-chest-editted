"""
AutoDL Connector - 连接和管理AutoDL服务器的工具包
"""

__version__ = "1.0.0"
__author__ = "AutoDL Connector Team"

from .src import AutoDLConnector, ConfigManager, FileTransfer, TaskManager

__all__ = [
    "AutoDLConnector",
    "ConfigManager",
    "FileTransfer",
    "TaskManager",
]