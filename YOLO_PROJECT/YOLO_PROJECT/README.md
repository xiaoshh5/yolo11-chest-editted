# yolo_project

这是一个 YOLO 训练环境模板（基于 `ultralytics` YOLO v8 API）的起始工程，放在 `/home/linsx/yolo_project`。

主要内容：
- `environment.yml`：用于通过 conda 创建环境（会用 pip 安装依赖）。
- `requirements.txt`：pip 依赖清单（`ultralytics`、`torch` 等）。
- `train_yolo.py`：训练脚本（调用 `ultralytics.YOLO` API）。
- `data/sample_data.yaml`：YOLO 数据配置模板（指向 `train/`、`val/` 目录）。
- `scripts/prepare_yolo_dataset.py`：简单的数据准备脚本（把数据组织成 YOLO 格式的 images/labels 目录结构）。

快速开始（推荐使用 conda）：

1) 创建并激活 conda 环境：

```bash
conda env create -f /home/linsx/yolo_project/environment.yml
conda activate yolo11
```

2) 准备数据：把你的训练/验证图像放到 ImageFolder 风格目录或直接按照 YOLO 格式放置。
   - 如果你的数据在 `/path/to/dataset` 且结构为 `train/images` 和 `val/images`，可以把路径写入 `data/sample_data.yaml`。
   - 否则，可用 `scripts/prepare_yolo_dataset.py` 做简单转换（请先阅读脚本注释）。

3) 训练示例：

```bash
python3 train_yolo.py \
  --data /home/linsx/yolo_project/data/sample_data.yaml \
  --epochs 50 \
  --batch 16 \
  --imgsz 640 \
  --project /home/linsx/yolo_project/runs
```

输出将会保存在 `--project` 路径下（ultralytics 默认格式）。

需要我为你：
- 运行 conda env 创建并安装依赖？
- 把你指定的数据集自动转换到 YOLO 格式并示范一次小规模训练（需要你授权在机器上运行）？

请告诉我你想要的下一步。
