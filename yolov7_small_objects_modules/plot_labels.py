import json
import cv2

# 檔案路徑
image_path = "kaggle_submission_datasets__/2018-10-04-Ampelkreuzungen_L0109.jpg"
json_path = "kaggle_submission_datasets__/2018-10-04-Ampelkreuzungen_L0109.json"
output_path = "annotated_image_cv2.png"

# 讀取圖片與 JSON
image = cv2.imread(image_path)
with open(json_path, 'r') as f:
    annotations = json.load(f)

# 字型設定
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_thickness = 1
text_color = (0, 140, 255)  # 橘色 BGR
box_color = (0, 140, 255)

for ann in annotations:
    x, y, w, h = map(int, ann["bbox"])
    score = ann.get("score", 0)
    label = ann.get("category_name", "object")

    text = f"{label} {score:.2f}"

    # 畫框
    cv2.rectangle(image, (x, y), (x + w, y + h), box_color, 1)

    # 放文字（先放背景再放字讓字清楚）
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    cv2.rectangle(image, (x, y - text_h - 4), (x + text_w, y), box_color, -1)
    cv2.putText(image, text, (x, y - 2), font, font_scale, (255, 255, 255), font_thickness, lineType=cv2.LINE_AA)

# 儲存圖片
cv2.imwrite(output_path, image)

print(f"圖片已儲存為：{output_path}")
