import argparse
import time
from pathlib import Path

import cv2
import torch.backends.cudnn as cudnn
import pandas as pd
from numpy import random
import numpy as np
import torch


from sahi_yolov7.SAHIYolo import SAHI_YoloV7
from cnn.CNN_Classifer import CNN_Classifier

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


import numpy as np


def iou(box1, box2):
    """ 計算兩個框的 IOU (交集比聯集) """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # 計算交集
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    # 計算各自的面積
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    # 計算 IOU
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


def box_center_distance(box1, box2):
    """ 計算兩個框的中心距離 """
    cx1, cy1 = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
    cx2, cy2 = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2
    return np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)


def merge_boxes(boxes1, boxes2, iou_threshold=0.2, distance_threshold=10):
    """
    保留 boxes2 的所有框，並加入 boxes1 中未與 boxes2 重疊的框。
    """
    filtered_boxes1 = []
    if len(boxes1) <= len(boxes2):
        boxes_right = boxes1
        boxes_left = boxes2
    else:
        boxes_right = boxes2
        boxes_left = boxes1

    for box1 in boxes_right:
        keep = True  # 預設保留 box1
        for box2 in boxes_left:
            if iou(box1, box2) > iou_threshold or box_center_distance(box1, box2) < distance_threshold:
                keep = False  # 發現重疊則捨棄
                break
        if keep:
            filtered_boxes1.append(box1)

    # 保留 boxes2 + 不重疊的 boxes1
    final_boxes = boxes_left + filtered_boxes1

    return [f"{box[0]} {box[1]} {box[2]} {box[3]}" for box in final_boxes]


def parse_boxes(box_list):
    """ 將字串形式的框轉換為數字 (x_min, y_min, x_max, y_max) """
    boxes = []
    for box in box_list:
        # 分割字串並處理每個部分，先轉換為 float 再轉換為 int
        box_values = box.split()
        box_values = [int(float(x.strip())) for x in box_values]
        boxes.append(tuple(box_values))
    return boxes


def convert_sahi_to_yolo(sahi_json, x_scale, y_scale, threshold, class_id=None):
    results = []
    boxes_results = []
    for pred_ in sahi_json:
        sahi_therehold = threshold * 1
        if pred_["score"] > sahi_therehold:
            if class_id is None or pred_["category_id"] == class_id:
                # 取得原始 bounding box
                x_min, y_min, width, height = pred_["bbox"]

                # 計算 x_max, y_max
                x_max = x_min + width
                y_max = y_min + height

                # 縮放座標 (如果需要)
                if x_scale != 1 or y_scale != 1:
                    x_min /= x_scale
                    y_min /= y_scale
                    x_max /= x_scale
                    y_max /= y_scale

                score = pred_["score"]
                category = class_id if class_id is not None else pred_["category_id"]
                results.append([x_min, y_min, x_max, y_max, score, category])
                boxes_results += [f'{x_min} {y_min} {x_max} {y_max}']

    # 將結果轉成 tensor
    # return [torch.tensor(results)] if results else [torch.empty(0, 6)]
    if len(results):
        results = torch.tensor(results, dtype=torch.float32)
    return results , boxes_results


def detect(save_img=False):
    boxes1 = []
    boxes2 = []
    all_results = []
    source, weights, view_img, save_txt, imgsz, trace, slice_mode = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace, opt.slice_mode
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    with open('output.csv', 'w') as f:
        f.write("ID,bbox\n")

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # slice_mode
        det_slice = []
        if slice_mode:
            sahi = SAHI_YoloV7(img_dir=path, model_path=opt.weights[0])
            sahi_pred_json, x_scale, y_scale = sahi.get_sliced_prediction_result()
            if len(opt.classes) == 1:
                det_slice, crd_slice = convert_sahi_to_yolo(sahi_json=sahi_pred_json, x_scale=x_scale, y_scale=y_scale, class_id=opt.classes[0], threshold=opt.conf_thres*1.2)
                boxes2 = parse_boxes(crd_slice)

        # Process detections
        crd = []
        for i, det in enumerate(pred):  # detections per image 這時img_size 還是你設定的3200
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            # if len(det_slice):
            #     det = torch.cat([det, det_slice], dim=0)

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                    x_min, y_min, x_max, y_max = xyxy
                    crd += [f'{x_min} {y_min} {x_max} {y_max}']

        boxes1 = parse_boxes(crd)
        crd_ = merge_boxes(boxes1=boxes1, boxes2=boxes2, iou_threshold=0.5, distance_threshold=40)
        print(f"origin model: {len(boxes1)}筆資料")
        print(f"sahi model: {len(boxes2)}筆資料")
        print(f"merge:{len(crd_)}")

        # 將結果記錄下來
        # cnn_model = CNN_Classifier(img_path=path, labels_list=crd_)
        # crd_ = cnn_model.classify()
        # print(f"final classify:{len(crd_)}")

        all_results.append({
            "name": p.name,
            "origin_model_count": len(boxes1),
            "sahi_count": len(boxes2),
            "merge_count": len(crd_)
        })
        df = pd.DataFrame(all_results)
        df.to_csv("detection_comparison.csv", index=False)

        with open(f'output.csv', 'a') as f:
            bbox = ' '.join(crd_) if (0 < len(crd_)) else '0'
            f.write(f"{p.name},{bbox}\n")
            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true',default=True, help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--slice_mode', action='store_true', default=True, help='using slice models')
    opt = parser.parse_args()
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
