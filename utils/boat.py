import cv2
import os
import numpy as np
from PIL import Image, ImageEnhance

# 設定輸入與輸出資料夾
input_img_dir = "../datasets/SkyFusion/final_train_dev/images/"  # 原始圖片資料夾
input_label_dir = "../datasets/SkyFusion/final_train_dev/annotations/"  # YOLO 標註資料夾
output_img_dir = "../datasets/SkyFusion/final_train_dev/images/"  # 儲存處理後圖片
output_label_dir = "../datasets/SkyFusion/final_train_dev/annotations/"  # 儲存標註

os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)


def process_image(image_path, label_path, output_img_path, output_label_path):
    # print(image_path)
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    with open(label_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:

        parts = line.strip().split()
        class_id = int(parts[0])
        x_center, y_center, bbox_width, bbox_height = map(float, parts[1:])

        if class_id == 1:  # 只處理船舶
            # 轉換 YOLO 格式為像素座標
            x1 = int((x_center - bbox_width / 2) * width)
            y1 = int((y_center - bbox_height / 2) * height)
            x2 = int((x_center + bbox_width / 2) * width)
            y2 = int((y_center + bbox_height / 2) * height)

            # **方法 1: 提高船的對比度**
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            enhancer = ImageEnhance.Contrast(pil_image)
            enhanced_image = enhancer.enhance(2.0)  # 增強對比度
            image = cv2.cvtColor(np.array(enhanced_image), cv2.COLOR_RGB2BGR)

            # **方法 2: 模糊背景**
            mask = np.zeros_like(image, dtype=np.uint8)
            mask[y1:y2, x1:x2] = image[y1:y2, x1:x2]
            blurred = cv2.GaussianBlur(image, (25, 25), 10)
            image = np.where(mask != 0, mask, blurred)

            # **方法 3: 使用 CLAHE 增強船的細節**
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            lab = cv2.merge((cl, a, b))
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        new_lines.append(line)

    # 儲存處理後的圖片與標註
    cv2.imwrite(output_img_path, image)
    with open(output_label_path, "w") as f:
        # print(f'{image}完成寫入')
        f.writelines(new_lines)

# 處理所有圖片與標註檔案


for file in os.listdir(input_label_dir):
    # print(file)
    if file.endswith(".txt"):
        image_file = file.replace(".txt", ".jpg")  # 假設圖片格式為 JPG
        image_path = os.path.join(input_img_dir, image_file)
        label_path = os.path.join(input_label_dir, file)
        output_img_path = os.path.join(output_img_dir, image_file)
        output_label_path = os.path.join(output_label_dir, file)
        if os.path.exists(image_path):  # 確保圖片存在
            process_image(image_path, label_path, output_img_path, output_label_path)
