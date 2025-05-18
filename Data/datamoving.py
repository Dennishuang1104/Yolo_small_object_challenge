import os
import shutil

# 原始目錄
source_dir = "xview-rawdata/images"  # 替換為實際目錄路徑
target_dir = "cnn_train_data/images"

# 確保目標目錄存在
os.makedirs(target_dir, exist_ok=True)

# 找出所有 txt 檔案
txt_files = [f for f in os.listdir(source_dir) if f.endswith(".txt")]

# 計算每個 txt 檔案的行數
file_line_counts = {}
for txt_file in txt_files:
    file_path = os.path.join(source_dir, txt_file)
    with open(file_path, "r", encoding="utf-8") as f:
        line_count = sum(1 for _ in f)
    file_line_counts[txt_file] = line_count

# 按行數排序並選擇前三個
top_txt_files = sorted(file_line_counts, key=file_line_counts.get, reverse=True)

# 複製前三個 txt 和對應的 jpg
for txt_file in top_txt_files:
    source_txt_path = os.path.join(source_dir, txt_file)
    target_txt_path = os.path.join(target_dir, txt_file)

    # 讀取前 3000 行並寫入新檔案
    with open(source_txt_path, "r", encoding="utf-8") as src, open(target_txt_path, "w", encoding="utf-8") as dst:
        for i, line in enumerate(src):
            if i >= 3000:
                break
            dst.write(line)

    # 對應的 jpg 檔案
    jpg_file = txt_file.replace(".txt", ".jpg")
    source_jpg_path = os.path.join(source_dir, jpg_file)
    target_jpg_path = os.path.join(target_dir, jpg_file)

    # 如果對應的 jpg 檔案存在，則複製
    if os.path.exists(source_jpg_path):
        shutil.copy2(source_jpg_path, target_jpg_path)



# 新增背景照片 + 新增建築物照片
print("已完成檔案篩選與複製！")