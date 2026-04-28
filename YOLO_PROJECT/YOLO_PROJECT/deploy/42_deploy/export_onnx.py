from ultralytics import YOLO

# Load a model
model = YOLO(r"C:\Users\Scm97\Desktop\tt100k\qq_3045834499\yolo11-tt100k\best.pt")  # load an official model
model.export(format="onnx")
# model = YOLO("path/to/best.pt")  # load a custom trained model

# Export the model
# model.export(format="onnx", dynamic=True, int8=True)


'''
format: 导出模型的目标格式（例如："......"）、 onnx, torchscript, tensorflow).
imgsz: 模型输入所需的图像大小（例如："......"）、 640 或 (height, width)).
half: 启用 FP16 量化，减少模型大小，并可能加快推理速度。
optimize: 针对移动或受限环境进行特定优化。
int8: 启用 INT8 量化，非常有利于边缘部署。
'''
# 模型得到的输出格式为（84x8400），84=边界框预测值4+数据集类别80， yolov8不另外对置信度预测， 而是采用类别里面最大的概率作为置信度score，8400是v8模型各尺度输出特征图叠加之后的结果（具体如何叠加可以看源码，一般推理不需要管）。本文对模型的输出进行如下操作，方便后处理：
# 安卓文件
# 安卓文件的版本选择2022.2.1