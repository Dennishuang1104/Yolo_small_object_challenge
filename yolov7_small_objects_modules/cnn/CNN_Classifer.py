import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os


class CNN_Classifier:
    def __init__(self, img_path, labels_list, model_path='cnn/cnn_224_revised.pth',
                 model_type='resnet18_modified', input_size=224):
        print(f"載入圖片: {img_path}")
        self.labels_list = labels_list
        self.img = cv2.imread(img_path)
        self.final_imgs_list = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size
        self.model_type = model_type

        # 根據輸入大小設置轉換
        # 使用高品質的上採樣方法
        if input_size == 224:
            self.transform = transforms.Compose([
                transforms.Resize((self.input_size, self.input_size), interpolation=Image.LANCZOS),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        self.model = self._load_model(model_path)

    def _load_model(self, model_path):
        # 根據模型類型創建對應架構
        if self.model_type == "resnet18_modified":
            model = self._create_modified_resnet18(2)  # 2 分類: car/no_car
        elif self.model_type == "mobilenetv2":
            model = self._create_mobilenetv2(2)
        elif self.model_type == "efficientnet":
            model = self._create_efficientnet(2)
        else:
            # 原始 ResNet18
            model = models.resnet18(weights=None)
            model.fc = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(model.fc.in_features, 2)
            )

        # 載入模型權重
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()  # 設置為推論模式
        return model

    def _create_modified_resnet18(self, num_classes):
        # 載入基礎模型
        model = models.resnet18(weights=None)

        # 修改第一層卷積層（從7x7改為3x3，並移除maxpool）
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # 修改前向傳播以跳過maxpool層
        original_forward = model.forward

        def new_forward(x):
            x = model.conv1(x)
            x = model.bn1(x)
            x = model.relu(x)
            # 跳過maxpool層
            # x = model.maxpool(x)

            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)

            x = model.avgpool(x)
            x = torch.flatten(x, 1)
            x = model.fc(x)

            return x

        model.forward = new_forward

        # 修改全連接層
        model.fc = nn.Sequential(
            nn.Dropout(0.3),  # 增加dropout比例
            nn.Linear(model.fc.in_features, num_classes)
        )

        return model

    def _create_mobilenetv2(self, num_classes):
        model = models.mobilenet_v2(weights=None)
        # 修改分類器
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.last_channel, num_classes)
        )
        return model

    def _create_efficientnet(self, num_classes):
        model = models.efficientnet_b0(weights=None)
        # 修改分類器
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )
        return model

    def pad_image(self, image, target_size=None):
        if target_size is None:
            target_size = (self.input_size, self.input_size)

        h, w = image.shape[:2]
        th, tw = target_size

        if h > th or w > tw:
            # 如果圖像太大，調整大小，但保持比例
            ratio = min(th / h, tw / w)
            new_h, new_w = int(h * ratio), int(w * ratio)
            image = cv2.resize(image, (new_w, new_h))

        h, w = image.shape[:2]
        top = max((th - h) // 2, 0)
        bottom = max(th - h - top, 0)
        left = max((tw - w) // 2, 0)
        right = max(tw - w - left, 0)

        # 使用黑色填充邊框
        return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    def classify(self, save_dir="output_imgs", confidence_threshold=0.5):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx, label in enumerate(self.labels_list):
            # 解析YOLO檢測框
            x_min, y_min, x_max, y_max = map(int, label.split())

            # 確保檢測框有效
            if x_min >= x_max or y_min >= y_max:
                print(f"無效的邊界框: {label}")
                continue

            # 裁剪圖像
            cropped_img = self.img[y_min:y_max, x_min:x_max]

            # 檢查裁剪圖像是否為空
            if cropped_img.size == 0:
                print(f"裁剪圖像為空: {label}")
                continue

            # 填充圖像到目標大小
            padded_img = self.pad_image(cropped_img)

            # 轉換到PIL格式
            padded_img_pil = Image.fromarray(cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB))

            # 準備輸入張量
            input_tensor = self.transform(padded_img_pil).unsqueeze(0).to(self.device)

            # 進行預測
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                predicted_label = torch.argmax(output, dim=1).item()
                confidence = probabilities[predicted_label].item()

            # 假設類別順序與訓練時相同 (0: car, 1: no_car)
            predicted_class = 'car' if predicted_label == 0 else 'no_car'

            # 創建相應的目錄
            class_dir = os.path.join(save_dir, predicted_class)
            os.makedirs(class_dir, exist_ok=True)

            # 保存圖像
            save_path = os.path.join(
                class_dir,
                f"{predicted_class}_{idx}_{x_min}_{y_min}_{x_max}_{y_max}_{confidence:.2f}.jpg"
            )
            cv2.imwrite(save_path, padded_img)

            # 只保存被分類為車輛的標籤
            if predicted_class == 'car' and confidence >= confidence_threshold:
                self.final_imgs_list.append(label)
                print(f"發現車輛，置信度: {confidence:.2f}，標籤: {label}")
            else:
                print(f"非車輛物體，置信度: {confidence:.2f}，標籤: {label}")

        return self.final_imgs_list


