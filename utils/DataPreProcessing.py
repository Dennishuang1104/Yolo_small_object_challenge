from PIL import Image
from tqdm import tqdm
import json
import os
import shutil
from pathlib import Path
import glob


class DataPreProcessing:
    def __init__(self, data_type):
        self.data_type = data_type
        self.default_category_list = ['Aircraft', 'ship', 'vehicle']
        self.img_type = 'jpg'
        self.file_type_list = ['images', 'labels']
        self.json_annotations_path = 'datasets/SkyFusion/annotations'
        self.dataset_path = f'datasets/SkyFusion/{data_type}'
        self.class_dict = {
            "Aircraft": 0,
            "ship": 1,
            "vehicle": 2
        }
        # 最終訓練的目錄
        self.final_train_path = 'datasets/SkyFusion/valid_dev'

        # 如果目錄不存在，則創建
        if not os.path.exists(self.final_train_path):
            os.makedirs(self.final_train_path)
            print(f"目錄 {self.final_train_path} 已建立")

    def transfer_json_information(self):
        # step 1 將照片複製到目標目錄
        original_train_data_img_path = f"{self.dataset_path}/images"
        final_train_data_img_path = f"{self.final_train_path}/images"

        if not os.path.exists(final_train_data_img_path):
            os.makedirs(final_train_data_img_path)
            print(f"目錄 {self.final_train_path} 已建立")

        for root, _, files in os.walk(original_train_data_img_path):
            print(root)
            for file in files:
                if file.endswith(".jpg"):
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(final_train_data_img_path, file)
                    shutil.copy2(src_path, dst_path)  # 保留 metadata

        # step 2 將json 標記資料轉換出來
        final_train_data_annotations_path = f"{self.final_train_path}/annotations"

        if not os.path.exists(final_train_data_annotations_path):
            os.makedirs(final_train_data_annotations_path)
            print(f"目錄 {final_train_data_annotations_path} 已建立")

        with open(self.json_annotations_path + f'/{self.data_type}.json', 'r') as j:
            json_data = json.load(j)

        imgs = {}
        for img in json_data['images']:
            imgs[img['id']] = {
                'file_name': Path(img['file_name']).stem,  # remove extension
                'height': img['height'],
                'width': img['width'],
            }

        for annotation in json_data['annotations']:
            attributes = imgs[annotation['image_id']]  # get image attributes by using image id
            category_id = int(annotation['category_id']) - 1  # zero-based, -1 to match the class id
            x, y, w, h = annotation['bbox']  # get bounding box information
            label = [category_id, ((2 * x + w) / (2 * attributes['width'])),
                     ((2 * y + h) / (2 * attributes['height'])), (w / attributes['width']),
                     (h / attributes['height'])]  # record and normalize

            file_name = attributes['file_name']
            with open(f'{final_train_data_annotations_path}/{file_name}.txt', 'a') as f:  # use 'a' for append
                f.write(' '.join(map(str, label)) + "\n")  # convert to string and write to file

    def process_traindata(self):
        self.transfer_json_information()
        len_ = 0
        dirs = ["annotations", "images"]
        # for directory in dirs:
        anno_dir = self.final_train_path + '/annotations'
        image_dir = anno_dir.replace('annotations', 'images')
        for filename in os.listdir(anno_dir):
            len_ += 1
            anno_file_path = os.path.join(anno_dir, filename)
            pic_path = anno_file_path.replace('annotations', 'images').replace('.txt', '.jpg')
            name, files_format = os.path.splitext(filename)
            # 設定新檔名(annotations)
            new_anno_filename = f"final_{self.data_type}_{len_}{files_format}"
            new_anno_path = os.path.join(anno_dir, new_anno_filename)
            os.rename(anno_file_path, new_anno_path)

            # 設定新檔名(images)
            new_image_filename = new_anno_filename.replace(files_format, '.jpg')
            new_image_path = os.path.join(image_dir, new_image_filename)
            os.rename(pic_path, new_image_path)