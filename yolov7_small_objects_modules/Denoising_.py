"""
降噪處理
"""

import cv2
import os
import glob


def denoise_nlm(image):
    """ 非局部均值降噪 (Non-Local Means Denoising) """
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)


def denoise_gaussian(image):
    """ 高斯模糊降噪 """
    return cv2.GaussianBlur(image, (5, 5), 1.5)


def denoise_bilateral(image):
    """ 雙邊濾波降噪 (Bilateral Filter) """
    return cv2.bilateralFilter(image, 9, 75, 75)


def denoise_median(image):
    """ 中值模糊降噪 (Median Blur) """
    return cv2.medianBlur(image, 5)

# 讀取圖片


# path = "data/SkyFusion/dev_test_1/2016-08-02-wolfsburg-rechts-R0060.jpg"
# image = cv2.imread(path)
#
#
# if image is None:
#     print(f"Error: Image not found at {path}")
# else:
#     # 選擇降噪方法
#     denoised_nlm = denoise_nlm(image)
#     denoised_bilateral = denoise_bilateral(image)
#
#     # 儲存降噪後的圖片
#     cv2.imwrite("data/SkyFusion/dev_test_1/2016-08-02-wolfsburg-rechts-R0060_denoised_nlm.jpg", denoised_nlm)
#     cv2.imwrite("data/SkyFusion/dev_test_1/2016-08-02-wolfsburg-rechts-R0060_denoised_bilateral.jpg", denoised_bilateral)
#
#     print("Denoised images saved successfully!")

# 設定目標目錄
target_directory = "../datasets/SkyFusion/final_train_denosing/images"

# 確保目錄存在
if not os.path.exists(target_directory):
    print(f"Error: Directory not found: {target_directory}")
else:
    # 取得所有圖片檔案 (支援 .jpg, .png, .jpeg)
    image_files = glob.glob(os.path.join(target_directory, "*.*g"))  # 匹配 jpg, png, jpeg

    if not image_files:
        print(f"No images found in {target_directory}")
    else:
        for image_path in image_files:
            image = cv2.imread(image_path)

            if image is None:
                print(f"Error: Failed to load {image_path}")
                continue  # 跳過無法讀取的圖片

            # 進行降噪
            denoised_nlm = denoise_nlm(image)
            denoised_bilateral = denoise_bilateral(image)

            # 生成新檔案名稱
            base_name = os.path.basename(image_path)  # 取得檔名
            name, ext = os.path.splitext(base_name)  # 拆分檔名與副檔名

            output_nlm = os.path.join(target_directory, f"{name}{ext}")
            output_bilateral = os.path.join(target_directory, f"{name}{ext}")

            # 儲存處理後的圖片
            cv2.imwrite(output_nlm, denoised_nlm)
            # cv2.imwrite(output_bilateral, denoised_bilateral)

            print(f"Denoised images saved: {output_nlm})")


