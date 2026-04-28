"""
文件传输模块 - 提供文件上传和下载功能
"""

import os
import paramiko
import logging
from pathlib import Path
from typing import Optional, List, Tuple
from tqdm import tqdm
import hashlib

logger = logging.getLogger(__name__)


class FileTransfer:
    """文件传输管理器"""
    
    def __init__(self, connector):
        """
        初始化文件传输管理器
        
        Args:
            connector: AutoDLConnector实例
        """
        self.connector = connector
        self.sftp = None
        
    def _ensure_sftp(self):
        """确保SFTP连接可用"""
        if not self.connector.is_connected:
            raise ConnectionError("未连接到服务器")
        
        if not self.sftp:
            self.sftp = self.connector.sftp
    
    def upload_file(self, local_path: str, remote_path: str, 
                   overwrite: bool = False, progress: bool = True) -> bool:
        """
        上传文件到服务器
        
        Args:
            local_path: 本地文件路径
            remote_path: 远程文件路径
            overwrite: 是否覆盖已存在的文件
            progress: 是否显示进度条
            
        Returns:
            bool: 上传是否成功
        """
        self._ensure_sftp()
        
        local_path = Path(local_path)
        if not local_path.exists():
            logger.error(f"本地文件不存在: {local_path}")
            return False
        
        try:
            # 检查远程文件是否存在
            try:
                self.sftp.stat(remote_path)
                remote_exists = True
            except FileNotFoundError:
                remote_exists = False
            
            if remote_exists and not overwrite:
                logger.warning(f"远程文件已存在: {remote_path}")
                return False
            
            # 获取文件大小
            file_size = local_path.stat().st_size
            
            # 上传文件
            with open(local_path, 'rb') as f:
                if progress:
                    with tqdm(total=file_size, unit='B', unit_scale=True, 
                             desc=f"上传 {local_path.name}") as pbar:
                        def callback(transferred, total):
                            pbar.update(transferred - pbar.n)
                        
                        self.sftp.putfo(f, remote_path, callback=callback)
                else:
                    self.sftp.putfo(f, remote_path)
            
            logger.info(f"文件上传成功: {local_path} -> {remote_path}")
            return True
            
        except Exception as e:
            logger.error(f"文件上传失败: {e}")
            return False
    
    def download_file(self, remote_path: str, local_path: str,
                     overwrite: bool = False, progress: bool = True) -> bool:
        """
        从服务器下载文件
        
        Args:
            remote_path: 远程文件路径
            local_path: 本地文件路径
            overwrite: 是否覆盖已存在的文件
            progress: 是否显示进度条
            
        Returns:
            bool: 下载是否成功
        """
        self._ensure_sftp()
        
        local_path = Path(local_path)
        
        try:
            # 检查本地文件是否存在
            if local_path.exists() and not overwrite:
                logger.warning(f"本地文件已存在: {local_path}")
                return False
            
            # 获取远程文件信息
            remote_attrs = self.sftp.stat(remote_path)
            file_size = remote_attrs.st_size
            
            # 下载文件
            with open(local_path, 'wb') as f:
                if progress:
                    with tqdm(total=file_size, unit='B', unit_scale=True,
                             desc=f"下载 {Path(remote_path).name}") as pbar:
                        def callback(transferred, total):
                            pbar.update(transferred - pbar.n)
                        
                        self.sftp.getfo(remote_path, f, callback=callback)
                else:
                    self.sftp.getfo(remote_path, f)
            
            logger.info(f"文件下载成功: {remote_path} -> {local_path}")
            return True
            
        except FileNotFoundError:
            logger.error(f"远程文件不存在: {remote_path}")
            return False
        except Exception as e:
            logger.error(f"文件下载失败: {e}")
            return False
    
    def upload_directory(self, local_dir: str, remote_dir: str,
                        overwrite: bool = False, progress: bool = True) -> Tuple[int, int]:
        """
        上传整个目录到服务器
        
        Args:
            local_dir: 本地目录路径
            remote_dir: 远程目录路径
            overwrite: 是否覆盖已存在的文件
            progress: 是否显示进度条
            
        Returns:
            Tuple[int, int]: (成功上传的文件数, 总文件数)
        """
        self._ensure_sftp()
        
        local_dir = Path(local_dir)
        if not local_dir.exists() or not local_dir.is_dir():
            logger.error(f"本地目录不存在: {local_dir}")
            return 0, 0
        
        # 获取所有文件
        all_files = []
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                file_path = Path(root) / file
                rel_path = file_path.relative_to(local_dir)
                all_files.append((file_path, rel_path))
        
        total_files = len(all_files)
        success_count = 0
        
        if progress:
            pbar = tqdm(all_files, desc="上传目录", unit="文件")
        else:
            pbar = all_files
        
        for local_file, rel_path in pbar:
            remote_file = str(Path(remote_dir) / rel_path)
            
            # 确保远程目录存在
            remote_parent = str(Path(remote_file).parent)
            try:
                self.sftp.stat(remote_parent)
            except FileNotFoundError:
                self._create_remote_directory(remote_parent)
            
            if self.upload_file(str(local_file), remote_file, overwrite, progress=False):
                success_count += 1
        
        logger.info(f"目录上传完成: {success_count}/{total_files} 个文件成功")
        return success_count, total_files
    
    def download_directory(self, remote_dir: str, local_dir: str,
                          overwrite: bool = False, progress: bool = True) -> Tuple[int, int]:
        """
        从服务器下载整个目录
        
        Args:
            remote_dir: 远程目录路径
            local_dir: 本地目录路径
            overwrite: 是否覆盖已存在的文件
            progress: 是否显示进度条
            
        Returns:
            Tuple[int, int]: (成功下载的文件数, 总文件数)
        """
        self._ensure_sftp()
        
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 获取远程目录下的所有文件
            all_files = self._list_remote_files(remote_dir)
        except FileNotFoundError:
            logger.error(f"远程目录不存在: {remote_dir}")
            return 0, 0
        
        total_files = len(all_files)
        success_count = 0
        
        if progress:
            pbar = tqdm(all_files, desc="下载目录", unit="文件")
        else:
            pbar = all_files
        
        for remote_file, rel_path in pbar:
            local_file = local_dir / rel_path
            
            # 确保本地目录存在
            local_file.parent.mkdir(parents=True, exist_ok=True)
            
            if self.download_file(remote_file, str(local_file), overwrite, progress=False):
                success_count += 1
        
        logger.info(f"目录下载完成: {success_count}/{total_files} 个文件成功")
        return success_count, total_files
    
    def _list_remote_files(self, remote_dir: str) -> List[Tuple[str, Path]]:
        """
        递归列出远程目录下的所有文件
        
        Args:
            remote_dir: 远程目录路径
            
        Returns:
            List[Tuple[str, Path]]: (远程文件路径, 相对路径) 列表
        """
        files = []
        
        def walk(current_dir, rel_path):
            try:
                items = self.sftp.listdir_attr(current_dir)
                for item in items:
                    item_path = f"{current_dir}/{item.filename}"
                    item_rel_path = rel_path / item.filename
                    
                    if item.st_mode & 0o40000:  # 目录
                        walk(item_path, item_rel_path)
                    else:  # 文件
                        files.append((item_path, item_rel_path))
            except Exception as e:
                logger.warning(f"无法访问目录 {current_dir}: {e}")
        
        walk(remote_dir, Path())
        return files
    
    def _create_remote_directory(self, remote_dir: str):
        """
        创建远程目录（包括父目录）
        
        Args:
            remote_dir: 远程目录路径
        """
        try:
            self.sftp.stat(remote_dir)
            return  # 目录已存在
        except FileNotFoundError:
            pass
        
        # 递归创建目录
        parts = Path(remote_dir).parts
        current_path = ""
        
        for part in parts:
            if not current_path:
                current_path = part
            else:
                current_path = f"{current_path}/{part}"
            
            try:
                self.sftp.stat(current_path)
            except FileNotFoundError:
                try:
                    self.sftp.mkdir(current_path)
                except Exception as e:
                    logger.warning(f"创建目录失败 {current_path}: {e}")
    
    def get_file_hash(self, file_path: str, algorithm: str = "md5") -> Optional[str]:
        """
        获取文件的哈希值
        
        Args:
            file_path: 文件路径
            algorithm: 哈希算法 (md5, sha1, sha256)
            
        Returns:
            Optional[str]: 哈希值，失败返回None
        """
        self._ensure_sftp()
        
        try:
            hash_func = getattr(hashlib, algorithm)()
            
            with self.sftp.open(file_path, 'rb') as f:
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        break
                    hash_func.update(chunk)
            
            return hash_func.hexdigest()
            
        except Exception as e:
            logger.error(f"获取文件哈希失败 {file_path}: {e}")
            return None
    
    def compare_files(self, local_path: str, remote_path: str, 
                     algorithm: str = "md5") -> bool:
        """
        比较本地文件和远程文件是否相同
        
        Args:
            local_path: 本地文件路径
            remote_path: 远程文件路径
            algorithm: 哈希算法
            
        Returns:
            bool: 文件是否相同
        """
        local_path = Path(local_path)
        if not local_path.exists():
            return False
        
        # 获取本地文件哈希
        try:
            hash_func = getattr(hashlib, algorithm)()
            with open(local_path, 'rb') as f:
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        break
                    hash_func.update(chunk)
            local_hash = hash_func.hexdigest()
        except Exception as e:
            logger.error(f"获取本地文件哈希失败 {local_path}: {e}")
            return False
        
        # 获取远程文件哈希
        remote_hash = self.get_file_hash(remote_path, algorithm)
        
        if remote_hash is None:
            return False
        
        return local_hash == remote_hash