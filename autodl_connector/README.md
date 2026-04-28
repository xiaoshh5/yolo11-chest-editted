# AutoDL Connector

一个用于连接和管理AutoDL服务器的Python工具包，提供SSH连接、文件传输、远程任务执行和配置管理功能。

## 功能特性

- **SSH连接管理**: 支持密码和SSH密钥认证
- **文件传输**: 支持文件/目录上传下载，带进度显示
- **远程命令执行**: 执行远程命令并获取结果
- **任务管理**: 提交、监控、取消远程任务
- **配置管理**: 管理多个服务器配置
- **命令行接口**: 提供便捷的CLI工具
- **上下文管理器**: 自动管理连接生命周期

## 安装

### 从源码安装

```bash
# 克隆项目
git clone https://github.com/yourusername/autodl-connector.git
cd autodl-connector

# 安装依赖
pip install -r requirements.txt

# 安装包
pip install -e .
```

### 依赖安装

```bash
pip install paramiko tqdm pyyaml
```

## 快速开始

### 1. 使用Python API

```python
from autodl_connector import AutoDLConnector

# 创建连接器
connector = AutoDLConnector(
    host="region-1.autodl.com",
    username="root",
    password="your_password"
)

# 连接服务器
if connector.connect():
    # 执行命令
    result = connector.execute_command("nvidia-smi")
    print(result['output'])
    
    # 获取服务器信息
    info = connector.get_server_info()
    print(info)
    
    # 断开连接
    connector.disconnect()
```

### 2. 使用上下文管理器

```python
from autodl_connector import AutoDLConnector

with AutoDLConnector(
    host="region-1.autodl.com",
    username="root",
    password="your_password"
) as connector:
    
    if connector.is_connected:
        # 执行多个命令
        result = connector.execute_command("hostname")
        print(f"主机名: {result['output']}")
```

### 3. 文件传输

```python
from autodl_connector import AutoDLConnector, FileTransfer

connector = AutoDLConnector(
    host="region-1.autodl.com",
    username="root",
    password="your_password"
)

if connector.connect():
    file_transfer = FileTransfer(connector)
    
    # 上传文件
    file_transfer.upload_file("local_file.txt", "/root/remote_file.txt")
    
    # 下载文件
    file_transfer.download_file("/root/remote_file.txt", "downloaded.txt")
    
    connector.disconnect()
```

### 4. 配置管理

```python
from autodl_connector import ConfigManager
from autodl_connector.src.config_manager import ServerConfig, AuthMethod

config_manager = ConfigManager()

# 添加服务器配置
server = ServerConfig(
    name="my_server",
    host="region-1.autodl.com",
    username="root",
    auth_method=AuthMethod.PASSWORD,
    password="your_password",
    description="我的训练服务器"
)

config_manager.add_server(server)

# 列出所有服务器
servers = config_manager.list_servers()
for server in servers:
    print(f"{server.name}: {server.host}")
```

## 命令行使用

### 配置管理

```bash
# 添加服务器配置
python -m autodl_connector.cli config add \
  --name my_server \
  --host region-1.autodl.com \
  --username root \
  --password your_password \
  --description "我的AutoDL服务器"

# 列出所有服务器
python -m autodl_connector.cli config list

# 显示服务器详情
python -m autodl_connector.cli config show --name my_server
```

### 连接服务器

```bash
# 连接到服务器
python -m autodl_connector.cli connect --server my_server

# 连接并显示服务器信息
python -m autodl_connector.cli connect --server my_server --info
```

### 文件传输

```bash
# 上传文件
python -m autodl_connector.cli upload \
  --server my_server \
  --local ./data.txt \
  --remote /root/data.txt

# 上传目录
python -m autodl_connector.cli upload \
  --server my_server \
  --local ./dataset \
  --remote /root/dataset

# 下载文件
python -m autodl_connector.cli download \
  --server my_server \
  --remote /root/results.txt \
  --local ./results.txt
```

### 执行命令

```bash
# 执行远程命令
python -m autodl_connector.cli exec \
  --server my_server \
  --command "nvidia-smi"

# 执行Python脚本
python -m autodl_connector.cli exec \
  --server my_server \
  --command "python train.py --epochs 10"
```

## 项目结构

```
autodl_connector/
├── __init__.py              # 包入口
├── cli.py                   # 命令行接口
├── src/                     # 源代码目录
│   ├── __init__.py
│   ├── connector.py         # SSH连接器
│   ├── file_transfer.py     # 文件传输
│   ├── task_manager.py      # 任务管理
│   └── config_manager.py    # 配置管理
├── examples/                # 使用示例
│   └── basic_usage.py
├── configs/                 # 配置文件目录
├── tests/                   # 测试文件
└── README.md                # 本文档
```

## 核心类说明

### AutoDLConnector
- `connect()`: 连接到服务器
- `disconnect()`: 断开连接
- `execute_command()`: 执行远程命令
- `get_server_info()`: 获取服务器信息
- `check_connection()`: 检查连接状态

### FileTransfer
- `upload_file()`: 上传单个文件
- `download_file()`: 下载单个文件
- `upload_directory()`: 上传整个目录
- `download_directory()`: 下载整个目录
- `compare_files()`: 比较文件是否相同

### TaskManager
- `submit_task()`: 提交远程任务
- `get_task()`: 获取任务信息
- `cancel_task()`: 取消任务
- `wait_for_task()`: 等待任务完成
- `monitor_tasks()`: 监控所有任务

### ConfigManager
- `add_server()`: 添加服务器配置
- `remove_server()`: 移除服务器配置
- `list_servers()`: 列出所有服务器
- `export_config()`: 导出配置到文件
- `import_config()`: 从文件导入配置

## 配置说明

配置文件存储在 `~/.autodl_connector/` 目录下：

- `servers.json`: 服务器配置
- `transfer_config.json`: 传输配置
- `task_config.json`: 任务配置

### 服务器配置示例

```json
{
  "servers": [
    {
      "name": "training_server",
      "host": "region-1.autodl.com",
      "port": 22,
      "username": "root",
      "auth_method": "password",
      "password": "your_password",
      "description": "GPU训练服务器",
      "tags": ["gpu", "training"]
    }
  ]
}
```

## 高级用法

### 批量任务执行

```python
from autodl_connector import AutoDLConnector, TaskManager

connector = AutoDLConnector(...)
task_manager = TaskManager(connector)

# 提交多个任务
tasks = [
    "python preprocess.py",
    "python train.py --epochs 50",
    "python evaluate.py"
]

task_ids = []
for cmd in tasks:
    task_id = task_manager.submit_task(cmd)
    task_ids.append(task_id)

# 等待所有任务完成
for task_id in task_ids:
    task = task_manager.wait_for_task(task_id)
    print(f"任务 {task_id} 完成: {task.status.value}")
```

### 断点续传

```python
from autodl_connector import AutoDLConnector, FileTransfer

connector = AutoDLConnector(...)
file_transfer = FileTransfer(connector)

# 检查文件是否已存在且相同
if not file_transfer.compare_files("local_file.txt", "/root/remote_file.txt"):
    # 文件不同，重新上传
    file_transfer.upload_file("local_file.txt", "/root/remote_file.txt", overwrite=True)
```

### 自定义回调

```python
from autodl_connector import AutoDLConnector, TaskManager

def task_callback(task):
    print(f"任务 {task.task_id} 状态: {task.status.value}")
    if task.status.value == "completed":
        print(f"输出: {task.result['output']}")
    elif task.status.value == "failed":
        print(f"错误: {task.error}")

connector = AutoDLConnector(...)
task_manager = TaskManager(connector)

# 提交带回调的任务
task_manager.submit_task(
    command="python long_running_task.py",
    callback=task_callback
)
```

## 注意事项

1. **安全性**: 密码以明文形式存储在配置文件中，建议使用SSH密钥认证
2. **网络稳定性**: 大文件传输时建议使用稳定的网络连接
3. **超时设置**: 根据任务复杂度合理设置超时时间
4. **资源限制**: 注意服务器的资源使用情况，避免过度占用

## 故障排除

### 连接失败
- 检查网络连接
- 验证服务器地址、端口、用户名和密码
- 确认防火墙设置

### 文件传输失败
- 检查文件权限
- 确认磁盘空间
- 验证网络稳定性

### 命令执行失败
- 检查命令语法
- 确认执行权限
- 查看错误输出

## 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 联系方式

- 项目地址: [https://github.com/yourusername/autodl-connector](https://github.com/yourusername/autodl-connector)
- 问题反馈: [Issues](https://github.com/yourusername/autodl-connector/issues)

## 更新日志

### v1.0.0 (2024-01-01)
- 初始版本发布
- 支持SSH连接和文件传输
- 提供命令行接口
- 实现配置管理功能