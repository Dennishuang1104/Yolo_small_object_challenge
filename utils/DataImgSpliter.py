import cv2
import os


def split_image_with_labels(image_path, label_path, output_dir, overlap=0.2, mode="horizontal"):
    """
    將圖片切割成兩半，並確保 20% 重疊區域，同時調整 YOLO 格式標註框的座標。

    參數:
    - image_path: 原始圖片路徑
    - label_path: YOLO 標註文件 (.txt) 路徑
    - output_dir: 存放切割後圖片和標註的資料夾
    - overlap: 重疊比例 (預設 20%)
    - mode: "horizontal" (左右切割) 或 "vertical" (上下切割)
    """
    # 讀取圖片
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"無法讀取圖片: {image_path}")

    H, W, _ = img.shape  # 取得原始圖片大小

    # 計算重疊區域
    if mode == "horizontal":  # 左右切割
        split_w = W // 2  # 先取一半
        overlap_w = int(W * overlap)  # 計算 20% 重疊區域
        left_img = img[:, :split_w + overlap_w]  # 取得左半部（含 20% 重疊區域）
        right_img = img[:, split_w - overlap_w:]  # 取得右半部（含 20% 重疊區域）

    elif mode == "vertical":  # 上下切割
        split_h = H // 2
        overlap_h = int(H * overlap)
        left_img = img[:split_h + overlap_h, :]  # 取得上半部（含 20% 重疊區域）
        right_img = img[split_h - overlap_h:, :]  # 取得下半部（含 20% 重疊區域）

    else:
        raise ValueError("mode 必須是 'horizontal' 或 'vertical'")

    # 產生輸出路徑
    base_name = os.path.basename(image_path).split('.')[0]  # 取得檔名
    os.makedirs(output_dir, exist_ok=True)

    # 儲存切割後的圖片
    left_img_path = os.path.join(output_dir + "/images", f"{base_name}_left.jpg")
    right_img_path = os.path.join(output_dir + "/images", f"{base_name}_right.jpg")
    cv2.imwrite(left_img_path, left_img)
    cv2.imwrite(right_img_path, right_img)

    # 讀取 YOLO 標註文件
    with open(label_path, "r") as f:
        lines = f.readlines()

    new_labels_left = []
    new_labels_right = []

    # 解析標註框並轉換
    for line in lines:
        parts = line.strip().split()
        cls_id = parts[0]  # 類別編號
        x_center, y_center, width, height = map(float, parts[1:])

        if mode == "horizontal":
            left_limit = (split_w + overlap_w) / W  # 左半部分的範圍
            right_limit = (split_w - overlap_w) / W  # 右半部分的範圍

            # 如果標註主要位於左側
            if x_center + width / 2 <= left_limit:
                new_x = x_center / left_limit  # 縮放 X 座標
                new_w = width / left_limit
                new_labels_left.append(f"{cls_id} {new_x:.6f} {y_center:.6f} {new_w:.6f} {height:.6f}\n")

            # 如果標註主要位於右側
            elif x_center - width / 2 >= right_limit:
                new_x = (x_center - right_limit) / (1 - right_limit)  # 縮放 X 座標
                new_w = width / (1 - right_limit)
                new_labels_right.append(f"{cls_id} {new_x:.6f} {y_center:.6f} {new_w:.6f} {height:.6f}\n")

        elif mode == "vertical":
            top_limit = (split_h + overlap_h) / H  # 上半部分的範圍
            bottom_limit = (split_h - overlap_h) / H  # 下半部分的範圍

            # 如果標註主要位於上側
            if y_center + height / 2 <= top_limit:
                new_y = y_center / top_limit  # 縮放 Y 座標
                new_h = height / top_limit
                new_labels_left.append(f"{cls_id} {x_center:.6f} {new_y:.6f} {width:.6f} {new_h:.6f}\n")

            # 如果標註主要位於下側
            elif y_center - height / 2 >= bottom_limit:
                new_y = (y_center - bottom_limit) / (1 - bottom_limit)  # 縮放 Y 座標
                new_h = height / (1 - bottom_limit)
                new_labels_right.append(f"{cls_id} {x_center:.6f} {new_y:.6f} {width:.6f} {new_h:.6f}\n")

    # 儲存新的標註文件
    left_label_path = os.path.join(output_dir + "/annotations" , f"{base_name}_left.txt")
    right_label_path = os.path.join(output_dir + "/annotations", f"{base_name}_right.txt")

    with open(left_label_path, "w") as f:
        f.writelines(new_labels_left)
    with open(right_label_path, "w") as f:
        f.writelines(new_labels_right)

    print(f"切割完成: {left_img_path}, {right_img_path}")
    print(f"標註完成: {left_label_path}, {right_label_path}")


def batch_split_images(input_images_dir, input_labels_dir, output_dir, overlap=0.2, mode="horizontal"):
    """
    遍歷 `input_images_dir` 目錄，對所有圖片執行 `split_image_with_labels()`，並將結果存入 `output_dir`。

    參數:
    - input_images_dir: 原始圖片資料夾（只包含圖片）
    - input_labels_dir: 標註資料夾（只包含 YOLO 格式的標註）
    - output_dir: 切割後的存放資料夾
    - overlap: 重疊比例 (預設 20%)
    - mode: "horizontal" (左右切割) 或 "vertical" (上下切割)
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir + '/images', exist_ok=True)
    os.makedirs(output_dir + '/annotations', exist_ok=True)

    for filename in os.listdir(input_images_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_images_dir, filename)
            label_path = os.path.join(input_labels_dir, filename.replace(".jpg", ".txt").replace(".png", ".txt"))
            split_image_with_labels(image_path, label_path, output_dir, overlap, mode)

    print("🚀 批次處理完成！")


batch_split_images(
    input_images_dir="../datasets/SkyFusion/final_train_v2_test/images",
    input_labels_dir="../datasets/SkyFusion/final_train_v2_test/annotations",
    output_dir="../datasets/SkyFusion/final_train_f1"
)

