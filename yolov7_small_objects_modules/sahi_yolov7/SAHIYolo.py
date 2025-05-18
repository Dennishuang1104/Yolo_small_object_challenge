from PIL import Image
from sahi.utils.yolov7 import download_yolov7_model
from sahi.predict import get_sliced_prediction, predict, get_prediction
from sahi import AutoDetectionModel
from pathlib import Path
import numpy as np
import json
import re
import os
import cv2


class SAHI_YoloV7:
    def __init__(self, img_dir: str, model_path=None):
        self.x_offset = None
        self.y_offset = None
        self._is_gpu = False
        self.conf_score = 0.25,
        self.device = "cpu",  # "cpu" or "cuda:0"
        self.category_mapping = {"vehicle": 2}
        self.trained_model_path = None
        if model_path:
            self.trained_model_path = model_path
        self.img_path = img_dir
        self.img_standard_size = (640, 640)
        self.resize_square = 1280   # 調整測試照片
        # self.img_size = 1280
        self.img_size = 640  # 輸入切割模型的圖片
        self.slice_size = 320  # 切割的大小
        self.image_extensions = {".jpg", ".jpeg", ".png"}
        self.width = 0
        self.height = 0
        # download_yolov7_model(self.trained_model_path)
        self.scale_x = 1
        self.scale_y = 1
        self._all_scale = 1
        self.imgs_files_name = ''
        self.resize_image_path = ''

    def resize_to_squire(self, image_path, size):
        """將圖片縮放並填充為 640x640，維持原比例"""
        resize_image_path = None
        image = cv2.imread(image_path)
        h, w = image.shape[:2]

        # 計算縮放比例，讓較長邊縮放至 size，保持原始比例
        self._all_scale = size / max(self.height, self.width)
        new_w = int(w * self._all_scale)
        new_h = int(h * self._all_scale)

        # 縮放影像
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 建立 size x size 的黑色背景（或白色、均值填充）
        canvas = 0 * np.ones((size, size, 3), dtype=np.uint8)  # 255 為白色，0 為黑色

        # 將縮放後的影像貼到中央
        self.y_offset = (size - new_h) // 2
        self.x_offset = (size - new_w) // 2
        canvas[self.y_offset:self.y_offset + new_h, self.x_offset : self.x_offset  + new_w] = resized_image

        # 儲存新圖片
        if image_path.casefold().endswith('jpg'):
            imgs_files_name = os.path.basename(image_path)
            resize_file_name = "1_resize_" + imgs_files_name
            resize_image_path = image_path.replace(imgs_files_name, resize_file_name)
            print(f"新增 resized: {resize_image_path}")

        if resize_image_path:
            cv2.imwrite(resize_image_path, canvas)
            self.resize_image_path = str(resize_image_path)

        return resize_image_path

    def evaluate_img_size(self):
        """
        checking using standard size (640) or 3072 (bigsize)
        :return:
        """
        # 取得目錄內所有檔案並排序 (確保每次都取相同的第一張圖)

        if os.path.isdir(self.img_path):
            image_files = sorted(
                [f for f in os.listdir(self.img_path) if os.path.splitext(f)[1].lower() in self.image_extensions]
                )

            if not image_files:
                print("目錄內沒有符合的圖片檔案。")
                return False

            first_image_path = os.path.join(self.img_path, image_files[0])

        else:
            first_image_path = self.img_path
        try:
            with Image.open(first_image_path) as img:
                size = img.size  # (width, height)
                print(f"圖片尺寸: {first_image_path}，尺寸: {size}")
                self.width, self.height = size
                self.resize_to_squire(image_path=first_image_path, size=self.resize_square)
                # if size > self.img_standard_size:
                #     self.resize_to_squire(image_path=first_image_path)
                #     self.img_size = 3072
                #     self.slice_size = 640
                #     self.scale_x = 1
                #     self.scale_y = 1
                #
                # if self.width % 32 != 0:
                #     # self.resize_to_multiple_of_32(image_path=first_image_path)

        except Exception as e:
            print(f"無法讀取圖片 {first_image_path}: {e}")
            return False

        finally:
            print(f"模型使用圖片尺寸： {self.resize_square}")

    def restore_json_bboxes(self, result_json, scale, pad_x, pad_y):
        """還原 COCO 格式 bbox 到原始圖片尺寸"""
        restored_results = []

        for item in result_json:
            x_min, y_min, w, h = item["bbox"]

            # 還原 bbox：去掉 padding 並除以 scale
            orig_x_min = int((x_min - pad_x) / scale)
            orig_y_min = int((y_min - pad_y) / scale)
            orig_w = int(w / scale)
            orig_h = int(h / scale)

            # 更新 bbox
            restored_item = item.copy()
            restored_item["bbox"] = [orig_x_min, orig_y_min, orig_w, orig_h]
            restored_results.append(restored_item)

        return restored_results

    def init_detection_model(self):
        if self._is_gpu:
            device = "cpu"
        else:
            device = 'cuda:0'
        detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov7hub',
            model_path=self.trained_model_path,
            confidence_threshold=0.25,
            device=device,  # or 'cuda:0,
            image_size=self.img_size,  # 原本是640
            category_mapping=self.category_mapping,
        )

        return detection_model

    def get_dir(self):
        base_dir = Path("runs")
        base_name = "detect_with_slice"
        existing_dirs = [d for d in base_dir.iterdir() if d.is_dir() and re.match(rf"^{base_name}\d*$", d.name)]

        if existing_dirs:
            max_num = max(
                int(d.name[len(base_name):]) if d.name[len(base_name):].isdigit() else 1
                for d in existing_dirs
            )
            new_dir_name = f"{base_name}{max_num + 1}"
        else:
            new_dir_name = base_name

        # 建立新目錄
        new_dir = base_dir / new_dir_name
        new_dir.mkdir(parents=True, exist_ok=True)
        return new_dir

    def get_sliced_prediction_result(self):
        self.evaluate_img_size()
        result_json = {}
        detection_model = self.init_detection_model()
        image_dir = Path(self.img_path)
        dir_name = self.get_dir()
        # 針對所有照片進行切割
        x = 0
        if os.path.isdir(self.img_path):
            # 如果是目錄，遍歷所有圖片檔案
            image_dir = image_dir.glob("**/*.[jJpP][pPnN]*")
        elif os.path.isfile(self.img_path):
            # 如果是單一檔案，處理該檔案
            image_dir = [image_dir]  # 包裝為列表，這樣可以統一處理
        for img_path in image_dir:
            resize_img_path = str(img_path).replace(img_path.name, f"1_resize_{img_path.name}")
            # if self.scale_x !=1 or self.scale_y!=1:
            # else:
            # resize_img_path = img_path
            # print(f"Processing: {img_path.name}")
            result = get_sliced_prediction(
                str(resize_img_path),
                detection_model,
                slice_height=self.slice_size,
                slice_width=self.slice_size,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
                postprocess_match_threshold=0.5
            )

            result_json = result.to_coco_annotations()

            # 處理Json
            if len(result_json) > 1:
                result_json = self.restore_json_bboxes(result_json=result_json,
                                                       scale=self._all_scale,
                                                       pad_x=self.x_offset,
                                                       pad_y=self.y_offset)

            # 如果有resize圖片 會在detect 處理
            result.export_visuals(export_dir=f"{dir_name}/")
            with open(f"{dir_name}/{img_path.stem}.json", "w", encoding="utf-8") as f:
                json.dump(result_json, f, ensure_ascii=False, indent=4)
                print(f'共偵測{len(result_json)} 物件')

        # 刪除resize 圖片
        if os.path.isfile(self.resize_image_path):
            os.remove(self.resize_image_path)
            print(f"✅ 已刪除: {self.resize_image_path}")

        return result_json, self.scale_x, self.scale_y
