import json
import cv2
import numpy as np
import os
import base64
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

import os.path as osp

from fontTools.subset import save_font

from tests.test_python import image


# 读取 Labelme 的 JSON 文件
def load_labelme_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


# 保存成 Labelme JSON 格式
def save_labelme_json(json_data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)


# 切割图片与 Mask
def slice_image_and_mask(image, mask, slice_size, stride):
    img_height, img_width = image.shape[:2]
    slices = []

    for y in range(0, img_height - slice_size + 1, stride):
        for x in range(0, img_width - slice_size + 1, stride):
            img_slice = image[y:y + slice_size, x:x + slice_size]
            mask_slice = mask[y:y + slice_size, x:x + slice_size]
            slices.append((img_slice, mask_slice, x, y))
    return slices


# 将图片转换为 Base64
def image_to_base64(image):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


# 将 Mask 转换为 Labelme JSON 格式，保留类别信息
def mask_to_labelme_json(mask, labelme_template, classes):
    annotations = []
    unique_classes = np.unique(mask)  # 获取掩码中的唯一类别
    for cls in unique_classes:
        if cls == 0:  # 假设0表示背景，不需要处理
            continue

        class_mask = np.zeros_like(mask, dtype=np.uint8)
        class_mask[mask == cls] = 255
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            points = [[float(point[0][0]), float(point[0][1])] for point in contour]

            # 添加标注
            annotations.append({
                "label": classes[cls],
                "points": points,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            })

    labelme_template['shapes'] = annotations
    return labelme_template


# 调整图片和掩码到切割大小的倍数
def pad_image_and_mask(image, mask, slice_size):
    img_height, img_width = image.shape[:2]

    # 计算需要的填充大小
    pad_height = (slice_size - img_height % slice_size) % slice_size
    pad_width = (slice_size - img_width % slice_size) % slice_size

    # 使用 cv2.copyMakeBorder 填充图片和掩码
    padded_image = cv2.copyMakeBorder(image, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    padded_mask = cv2.copyMakeBorder(mask, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=0)

    return padded_image, padded_mask


# 主函数
def process_image_and_mask( suffix_name, image_path, json_path, output_dir, slice_size=256, stride=128):
    # 读取图片和 JSON
    image = cv2.imread(image_path)
    image_height, image_width = image.shape[:2]
    labelme_data = load_labelme_json(json_path)

    # 检查 JSON 中的图像宽度和高度
    if 'imageWidth' not in labelme_data or 'imageHeight' not in labelme_data:
        labelme_data['imageWidth'] = image_width
        labelme_data['imageHeight'] = image_height

    # 生成初始 Mask
    classes = {}
    index = 1
    mask = np.zeros((image_height, image_width), dtype=np.uint16)
    for shape in labelme_data['shapes']:
        points = np.array(shape['points'], dtype=np.int32)
        classes[index] = shape["label"]
        # 使用index的数值对图像进行填充的操作，将原先的points的区域使用不同的id进行实例的区分，同时在上面的字典中带有原始的类别标记。
        cv2.fillPoly(mask, [points], index)
        index += 1

    # 调整图片和掩码的大小，注意，此时的mask的数值已经是超出了0-255的范围
    padded_image, padded_mask = pad_image_and_mask(image, mask, slice_size)

    # 切割图片和 Mask，此时的mask是切割之后的结果，上面的操作只是做切割的作用
    slices = slice_image_and_mask(padded_image, padded_mask, slice_size, stride)

    # 遍历切割后的片段，生成新的 Labelme JSON 和切割图片
    for i, (img_slice, mask_slice, x_offset, y_offset) in enumerate(slices):
        slice_img_path = os.path.join(output_dir, f"{suffix_name}_{i}.png")
        slice_json_path = os.path.join(output_dir, f"{suffix_name}_{i}.json")

        # 保存切割后的图片
        cv2.imwrite(slice_img_path, img_slice)

        # 转换 mask 为 labelme 的 JSON，并保留类别信息, 这个位置传入的是生成的mask图像，此时的mask_slice是超过了255数值内容的东西 man
        json_slice = mask_to_labelme_json(mask_slice, labelme_data.copy(), classes)

        # 更新 JSON 中的 imagePath 和 imageData
        json_slice['imagePath'] = os.path.basename(slice_img_path)
        json_slice['imageData'] = image_to_base64(img_slice)

        # 更新 JSON 中的图像宽度和高度
        json_slice['imageWidth'] = img_slice.shape[1]
        json_slice['imageHeight'] = img_slice.shape[0]

        # 保存新的 JSON 文件
        save_labelme_json(json_slice, slice_json_path)


if __name__ == "__main__":
    import time
    # 获取当前时间戳
    current_timestamp = time.time()
    print(int(current_timestamp))


    src_folder = ""
    save_folder = ""
    image_names =  os.listdir(src_folder)
    for image_name in image_names:
        print(f"当前处理的图像为：{image_name}")
        suffix_name = image_name.split(".")[0]
        # name = "IMG_1046"
        image_path = osp.join(src_folder, suffix_name + '.JPG')
        json_path = osp.join(src_folder, suffix_name + '.json')
        output_dir = osp.join(save_folder, 'output_' + suffix_name)
        os.makedirs(output_dir, exist_ok=True)
        slice_size = 1024  # 切片大小
        stride = 300  # 间隔

        process_image_and_mask(suffix_name, image_path, json_path, output_dir, slice_size,stride)
