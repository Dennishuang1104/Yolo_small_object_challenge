from PIL import Image
from sahi.utils.yolov7 import download_yolov7_model
from sahi.predict import get_sliced_prediction, predict, get_prediction
from sahi import AutoDetectionModel
import json

# download yolov7 model
yolov7_model_path = '../runs/train/exp/weights/054_best.pt'
download_yolov7_model(yolov7_model_path)

# download test images into demo_data folder
# download_from_url('https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/small-vehicles1.jpeg', 'demo_data/small-vehicles1.jpeg')

detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov7hub',  # or 'yolov7hub'
    model_path=yolov7_model_path,
    confidence_threshold=0.25,
    device="cpu",  # or 'cuda:0,
    image_size=3072,  # '
    category_mapping={"vehicle": 2},
    # category_remapping={"vehicle": 2}
)

result = get_sliced_prediction(
    "../data/SkyFusion/test_kaggle/2006-05-03-Allianz-links-yr7e0006.jpg",
    detection_model,
    slice_height=1280,
    slice_width=1280,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2
)

result.export_visuals(export_dir="../runs/detect/dev/")
result_json = result.to_coco_annotations()
result.to_coco_predictions(image_id='dev')

with open("../runs/detect/dev/1280_1_img/output2.json", "w", encoding="utf-8") as f:
    json.dump(result_json, f, ensure_ascii=False, indent=4)

print(len(result_json))
print("JSON 已輸出到 output.json")

# 1790
