import os
import cv2


def video2img(video_folder, img_folder, frequency=10):
    video_files = os.listdir(video_folder)
    for video_file in video_files:
        print("{}图片提取中...".format(video_file))
        times = 0
        img_idx = 0
        video_file_name = video_file.split(".")[0]
        video_file_path = os.path.join(video_folder, video_file)
        img_sub_folder = os.path.join(img_folder, video_file_name)
        if not os.path.exists(img_sub_folder):
            os.makedirs(img_sub_folder)
            camera = cv2.VideoCapture(video_file_path)
            while True:
                times += 1
                res, image = camera.read()
                if not res:
                    print('EOF')
                    break
                if times % frequency == 0:
                    img_name = video_file_name + "_" + str(img_idx) + ".jpg"
                    img_idx = img_idx + 1
                    img_save_path = os.path.join(img_sub_folder, img_name)
                    cv2.imwrite(img_save_path, image)
            print('{}文件图片提取结束：{}张'.format(video_file, img_idx))
            camera.release()


if __name__ == '__main__':
    video2img(video_folder="F:/datas/tmp/shuju/shuju/shuju", img_folder="F:/datas/tmp/shuju/shuju/train", frequency=10)
