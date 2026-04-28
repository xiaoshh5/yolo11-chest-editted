from PIL import Image, ImageDraw
import os

def make_dataset(root='/home/linsx/yolo_project/data/sample_dataset', imgs_train=4, imgs_val=2, w=320, h=240):
    os.makedirs(root, exist_ok=True)
    for split, n in [('train', imgs_train), ('val', imgs_val)]:
        img_dir = os.path.join(root, split, 'images')
        lbl_dir = os.path.join(root, split, 'labels')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        for i in range(n):
            img_path = os.path.join(img_dir, f'{split}_{i}.jpg')
            # create simple image with colored rectangle
            im = Image.new('RGB', (w,h), (240,240,240))
            draw = ImageDraw.Draw(im)
            # alternate classes 0 and 1
            cls = i % 2
            # rectangle coords
            if cls == 0:
                box = (30,30,120,120)
                color = (255,0,0)
            else:
                box = (180,60,280,200)
                color = (0,0,255)
            draw.rectangle(box, outline=color, width=4)
            im.save(img_path, quality=85)
            # write YOLO label: class x_center y_center w h (normalized)
            x1,y1,x2,y2 = box
            x_c = (x1+x2)/2.0 / w
            y_c = (y1+y2)/2.0 / h
            ww = (x2-x1)/w
            hh = (y2-y1)/h
            lbl_path = os.path.join(lbl_dir, f'{split}_{i}.txt')
            with open(lbl_path, 'w') as f:
                f.write('{} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(cls, x_c, y_c, ww, hh))
    print('Created sample dataset at', root)

if __name__ == '__main__':
    make_dataset()
