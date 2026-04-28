import os

txt_folder = r"D:\1\demo\data\labels\train"  # txt文件所在的文件夹路径

# 遍历txt文件列表
for txt_file in os.listdir(txt_folder):
    if txt_file.endswith(".txt"):
        txt_path = os.path.join(txt_folder, txt_file)
        with open(txt_path, "r") as f:
            lines = f.readlines()

        # 修改类别索引为0
        modified_lines = []
        for line in lines:
            line = line.strip().split()
            line[0] = "0"  # 将类别索引修改为0
            modified_lines.append(" ".join(line))

        # 将修改后的内容写回txt文件
        with open(txt_path, "w") as f:
            f.write("\n".join(modified_lines))