import os
from ultralytics import YOLO

mode = 'val' # train/val
# Load a model
try:
    if mode == 'train':
        model = YOLO('yolov8m-det.yaml').load(r"yolov8m-det.pt")  # build a new model from YAML
        results = model.train(cfg = "cfg_detect.yaml", data='mine_detect.yaml', epochs=200, imgsz=640)
        with open(os.path.join(model.trainer.args.save_dir, "log_info.txt"), "w+") as f:
            log_info = "2024/03/12 17:47 版训练, 扩增模型为yolov8m, 同时加载预训练模型进行训练"
            f.write(log_info)
        f.close()
    elif mode == 'val':
        model_name = "test_detect_0312_1546"
        save_name = "val_detect_0312_1546"
        save_root = "detect"
        model = YOLO(os.path.join(save_root, model_name,'weights','best.pt'))  # load an official model
        metrics = model.val(save_dir = os.path.join(save_root, save_name), cfg = "cfg_detect.yaml", data='mine_detect.yaml')  # no arguments needed, dataset and settings remembered
        with open(os.path.join(save_root, save_name, "log_info.txt"), "w+") as f:
            log_info = "test_detect_0312_1546"
            f.write(log_info)
        f.close()
        print(f"map50-95:{metrics.box.map}")
except Exception as e:
    print(e)

# cfg = "cfg_detect.yaml", 
# Validate the model
# model = YOLO('/media/libinWorkSpace/ultralytics/runs/detect/train2/weights/best.pt')  # load an official model
# metrics = model.val()  # no arguments needed, dataset and settings remembered
# metrics.box.app_ap[:, :]  # 17 * 10, 10(50,55,60,65,70,75,80,85,90,95)
# metrics.box.map    # map50-95
# metrics.box.map50  # map50
# metrics.box.map75  # map75 
# metrics.box.maps   # a list contains map50-95 of each category