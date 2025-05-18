import os
import pandas as pd


import os
import pandas as pd

import os
import pandas as pd


import os
import pandas as pd


def find_and_remove_invalid_txt_files(directory, output_csv):
    invalid_files = []

    # 遍歷目錄中的所有文件
    for filename in os.listdir(directory):
        print(filename)
        filepath = os.path.join(directory, filename)

        # 確保是 .txt 文件，且是普通文件（排除資料夾）
        if filename.endswith(".txt") and os.path.isfile(filepath):
            try:
                # 檢查文件是否為空
                if os.stat(filepath).st_size == 0:
                    invalid_files.append({"Filename": filename, "Path": filepath, "Reason": "Empty file"})
                    os.remove(filepath)
                    print(f"已刪除空文件: {filename}")
                    continue

                with open(filepath, "r", encoding="utf-8") as file:
                    first_line = file.readline().strip()  # 讀取第一行並去除前後空白

                # 檢查第一行是否符合特定條件
                if first_line and first_line.split()[0] == "1":
                    invalid_files.append({"Filename": filename, "Path": filepath, "Reason": "Starts with 1"})
                    os.remove(filepath)
                    print(f"已刪除: {filename}")

                elif first_line and first_line.split()[0] != "2":
                    invalid_files.append({"Filename": filename, "Path": filepath, "Reason": "Does not start with 2"})
                    os.remove(filepath)
                    print(f"已刪除: {filename}")

            except Exception as e:
                print(f"讀取 {filename} 時發生錯誤: {e}")

    # 如果有不合格的文件，則存成 CSV
    if invalid_files:
        df = pd.DataFrame(invalid_files)
        df.to_csv(output_csv, index=False)
        print(f"已將結果儲存至 {output_csv}")
    else:
        print("沒有發現需要刪除的 .txt 文件。")

    # 檢查目錄內是否還有 .txt 文件，如果沒有則刪除目錄
    remaining_txt_files = [f for f in os.listdir(directory) if f.endswith(".txt")]
    if not remaining_txt_files:
        try:
            os.rmdir(directory)
            print(f"目錄 {directory} 已刪除，因為所有 .txt 檔案都無內容")
        except Exception as e:
            print(f"刪除目錄 {directory} 時發生錯誤: {e}")


# # 設定目錄路徑（請修改成你的目錄）
directory_path = '../datasets/SkyFusion/final_train_onlycar/annotations/'  # 請替換為你的目錄路徑
output_csv_path = "empty_files.csv"  # 請替換成儲存 CSV 的路徑

# 執行函式
find_and_remove_invalid_txt_files(directory_path, output_csv_path)


def delete_files_from_csv(csv_path, target_directory):
    # 讀取 CSV 檔案
    df = pd.read_csv(csv_path)

    # 確保 CSV 檔案有 "Filename" 欄位
    if "Filename" not in df.columns:
        print("CSV 檔案缺少 'Filename' 欄位，請檢查格式。")
        return

    deleted_files = []  # 紀錄成功刪除的文件
    not_found_files = []  # 紀錄未找到的文件

    for filename in df["Filename"]:
        # 構造要刪除的文件路徑
        annotation_file = os.path.join(target_directory, "annotations", filename)
        image_file = os.path.join(target_directory, "images", os.path.splitext(filename)[0] + ".jpg")

        for file_path in [annotation_file, image_file]:
            if os.path.exists(file_path):
                os.remove(file_path)  # 刪除文件
                deleted_files.append(file_path)
            else:
                not_found_files.append(file_path)

    # 列印刪除結果
    print(f"成功刪除 {len(deleted_files)} 個文件")
    if not_found_files:
        print(f"有 {len(not_found_files)} 個文件未找到")


# 設定 CSV 路徑 & 目標目錄
csv_path = "empty_files.csv"  # 請替換為你的 CSV 路徑
target_directory = '../datasets/SkyFusion/final_train_v1'   # 請替換為你的目標資料夾
# 執行刪除
delete_files_from_csv(csv_path, target_directory)
