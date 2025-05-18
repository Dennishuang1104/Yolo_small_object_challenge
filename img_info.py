import os
import cv2
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm


def analyze_image_properties(image_path):
    # 讀圖（PIL 和 OpenCV 各自取所需）
    img_pil = Image.open(image_path).convert("RGB")
    img_cv = cv2.imread(image_path)

    # 圖片大小
    width, height = img_pil.size

    # 光度（灰階平均）
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)

    # 對比度（灰階標準差）
    contrast = np.std(gray)

    return {
        "filename": os.path.basename(image_path),
        "width": width,
        "height": height,
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2)
    }


def analyze_directory_images(image_dir):
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(image_extensions)]

    records = []
    for img_path in tqdm(image_files, desc="Analyzing images"):
        try:
            record = analyze_image_properties(img_path)
            records.append(record)
        except Exception as e:
            print(f"❌ Failed to process {img_path}: {e}")

    df = pd.DataFrame(records)
    return df


# 使用範例
if __name__ == "__main__":
    image_directory = "yolov7_pytorch_modules/kaggle_submission_datasets"  # 修改這裡
    df_result = analyze_directory_images(image_directory)

    # 顯示統計摘要
    print(df_result.describe())

    # 儲存結果
    df_result.to_csv("image_analysis_report.csv", index=False)