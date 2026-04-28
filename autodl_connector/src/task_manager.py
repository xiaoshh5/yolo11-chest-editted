"""
任务管理器 - 管理远程任务执行和监控
"""

import time
import threading
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from enum import Enum
import json

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class RemoteTask:
    """远程任务类"""
    
    def __init__(self, task_id: str, command: str, 
                 timeout: int = 3600, working_dir: Optional[str] = None):
        """
        初始化远程任务
        
        Args:
            task_id: 任务ID
            command: 要执行的命令
            timeout: 超时时间（秒）
            working_dir: 工作目录
        """
        self.task_id = task_id
        self.command = command
        self.timeout = timeout
        self.working_dir = working_dir
        
        self.status = TaskStatus.PENDING
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.progress: float = 0.0
        self.output: List[str] = []
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'task_id': self.task_id,
            'command': self.command,
            'timeout': self.timeout,
            'working_dir': self.working_dir,
            'status': self.status.value,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'result': self.result,
            'error': self.error,
            'progress': self.progress,
            'output': self.output
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RemoteTask':
        """从字典创建"""
        task = cls(
            task_id=data['task_id'],
            command=data['command'],
            timeout=data.get('timeout', 3600),
            working_dir=data.get('working_dir')
        )
        
        task.status = TaskStatus(data['status'])
        task.start_time = datetime.fromisoformat(data['start_time']) if data['start_time'] else None
        task.end_time = datetime.fromisoformat(data['end_time']) if data['end_time'] else None
        task.result = data.get('result')
        task.error = data.get('error')
        task.progress = data.get('progress', 0.0)
        task.output = data.get('output', [])
        
        return task


class TaskManager:
    """任务管理器"""
    
    def __init__(self, connector):
        """
        初始化任务管理器
        
        Args:
            connector: AutoDLConnector实例
        """
        self.connector = connector
        self.tasks: Dict[str, RemoteTask] = {}
        self.task_threads: Dict[str, threading.Thread] = {}
        self.running = True
        
    def submit_task(self, command: str, task_id: Optional[str] = None,
                   timeout: int = 3600, working_dir: Optional[str] = None,
                   callback: Optional[Callable[[RemoteTask], None]] = None) -> str:
        """
        提交任务
        
        Args:
            command: 要执行的命令
            task_id: 任务ID，如果为None则自动生成
            timeout: 超时时间（秒）
            working_dir: 工作目录
            callback: 任务完成时的回调函数
            
        Returns:
            str: 任务ID
        """
        if task_id is None:
            task_id = f"task_{int(time.time())}_{len(self.tasks)}"
        
        if task_id in self.tasks:
            raise ValueError(f"任务ID已存在: {task_id}")
        
        task = RemoteTask(task_id, command, timeout, working_dir)
        self.tasks[task_id] = task
        
        # 启动任务线程
        thread = threading.Thread(
            target=self._execute_task,
            args=(task, callback),
            daemon=True
        )
        self.task_threads[task_id] = thread
        thread.start()
        
        logger.info(f"任务已提交: {task_id} - {command}")
        return task_id
    
    def _execute_task(self, task: RemoteTask, callback: Optional[Callable[[RemoteTask], None]] = None):
        """执行任务（在线程中运行）"""
        task.status = TaskStatus.RUNNING
        task.start_time = datetime.now()
        
        try:
            # 如果有工作目录，先切换到该目录
            full_command = task.command
            if task.working_dir:
                full_command = f"cd {task.working_dir} && {task.command}"
            
            # 执行命令
            result = self.connector.execute_command(full_command, timeout=task.timeout)
            
            task.result = result
            task.output = result['output'].split('\n') if result['output'] else []
            
            if result['success']:
                task.status = TaskStatus.COMPLETED
                task.progress = 1.0
                logger.info(f"任务完成: {task.task_id}")
            else:
                task.status = TaskStatus.FAILED
                task.error = result['error']
                logger.error(f"任务失败: {task.task_id} - {result['error']}")
                
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            logger.error(f"任务执行异常: {task.task_id} - {e}")
            
        finally:
            task.end_time = datetime.now()
            
            # 执行回调函数
            if callback:
                try:
                    callback(task)
                except Exception as e:
                    logger.error(f"回调函数执行失败: {e}")
    
    def get_task(self, task_id: str) -> Optional[RemoteTask]:
        """
        获取任务信息
        
        Args:
            task_id: 任务ID
            
        Returns:
            Optional[RemoteTask]: 任务对象，如果不存在返回None
        """
        return self.tasks.get(task_id)
    
    def get_all_tasks(self) -> List[RemoteTask]:
        """
        获取所有任务
        
        Returns:
            List[RemoteTask]: 任务列表
        """
        return list(self.tasks.values())
    
    def cancel_task(self, task_id: str) -> bool:
        """
        取消任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            bool: 是否成功取消
        """
        task = self.tasks.get(task_id)
        if not task:
            logger.warning(f"任务不存在: {task_id}")
            return False
        
        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            logger.warning(f"任务已结束，无法取消: {task_id}")
            return False
        
        # 尝试终止远程进程（需要实现）
        # 这里可以发送SIGTERM或SIGKILL信号
        try:
            # 查找并终止相关进程
            kill_command = f"pkill -f '{task.command}'"
            self.connector.execute_command(kill_command, timeout=5)
        except:
            pass
        
        task.status = TaskStatus.CANCELLED
        task.end_time = datetime.now()
        task.error = "任务被取消"
        
        logger.info(f"任务已取消: {task_id}")
        return True
    
    def wait_for_task(self, task_id: str, timeout: Optional[int] = None, 
                     poll_interval: float = 1.0) -> Optional[RemoteTask]:
        """
        等待任务完成
        
        Args:
            task_id: 任务ID
            timeout: 等待超时时间（秒）
            poll_interval: 轮询间隔（秒）
            
        Returns:
            Optional[RemoteTask]: 任务对象，如果超时返回None
        """
        task = self.tasks.get(task_id)
        if not task:
            return None
        
        start_time = time.time()
        
        while self.running:
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.TIMEOUT]:
                return task
            
            if timeout and (time.time() - start_time) > timeout:
                task.status = TaskStatus.TIMEOUT
                task.end_time = datetime.now()
                task.error = "等待超时"
                return task
            
            time.sleep(poll_interval)
        
        return None
    
    def monitor_tasks(self, interval: float = 5.0):
        """
        监控所有任务（在单独线程中运行）
        
        Args:
            interval: 监控间隔（秒）
        """
        def monitor():
            while self.running:
                try:
                    self._check_timeout_tasks()
                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"任务监控异常: {e}")
                    time.sleep(interval)
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
    
    def _check_timeout_tasks(self):
        """检查超时任务"""
        current_time = time.time()
        
        for task_id, task in self.tasks.items():
            if task.status == TaskStatus.RUNNING and task.start_time:
                elapsed = current_time - task.start_time.timestamp()
                if elapsed > task.timeout:
                    task.status = TaskStatus.TIMEOUT
                    task.end_time = datetime.now()
                    task.error = f"任务执行超时 ({task.timeout}秒)"
                    
                    # 尝试终止进程
                    try:
                        kill_command = f"pkill -f '{task.command}'"
                        self.connector.execute_command(kill_command, timeout=5)
                    except:
                        pass
                    
                    logger.warning(f"任务超时: {task_id}")
    
    def save_tasks(self, filepath: str):
        """
        保存任务到文件
        
        Args:
            filepath: 文件路径
        """
        tasks_data = {
            'tasks': [task.to_dict() for task in self.tasks.values()]
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(tasks_data, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"任务已保存到: {filepath}")
        except Exception as e:
            logger.error(f"保存任务失败: {e}")
    
    def load_tasks(self, filepath: str):
        """
        从文件加载任务
        
        Args:
            filepath: 文件路径
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                tasks_data = json.load(f)
            
            for task_data in tasks_data.get('tasks', []):
                task = RemoteTask.from_dict(task_data)
                self.tasks[task.task_id] = task
            
            logger.info(f"任务已从文件加载: {filepath}")
        except Exception as e:
            logger.error(f"加载任务失败: {e}")
    
    def clear_completed_tasks(self):
        """清理已完成的任务"""
        tasks_to_remove = []
        
        for task_id, task in self.tasks.items():
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.TIMEOUT]:
                # 检查是否超过保留时间（例如24小时）
                if task.end_time:
                    elapsed = (datetime.now() - task.end_time).total_seconds()
                    if elapsed > 24 * 3600:  # 24小时
                        tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            del self.tasks[task_id]
            if task_id in self.task_threads:
                del self.task_threads[task_id]
        
        if tasks_to_remove:
            logger.info(f"已清理 {len(tasks_to_remove)} 个已完成的任务")
    
    def stop(self):
        """停止任务管理器"""
        self.running = False
        
        # 取消所有运行中的任务
        for task_id, task in self.tasks.items():
            if task.status == TaskStatus.RUNNING:
                self.cancel_task(task_id)
        
        logger.info("任务管理器已停止")
    
    def __del__(self):
        """析构函数"""
        self.stop()