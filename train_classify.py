import os
import os.path as osp
import PIL.Image as Image
from ultralytics import YOLO
mode = 'test'
# train
try:
    # model = YOLO("yolov8m-cls.yaml")  # build a new model from scratch
    if mode == 'train':
        model = YOLO("yolov8m-cls.yaml")#.load(r"/media/libinWorkSpace/ultralytics/yolov8m-cls.pt")  # build a new model from scratch
        model.train(cfg = "cfg_classify.yaml", data="/media/libinWorkSpace/YZK_FuBu/cls_dataset")  # train the model
        with open(os.path.join(model.trainer.args.save_dir, "log_info.txt"), "w+") as f:
            log_info = "train_BCELoss_0318_1357 使用BCELoss  利用v0版本数据集"
            f.write(log_info)
        f.close()
    elif mode == 'val':
        model_name = "test_031217"
        save_name = "val_031217"
        save_root = "classify"
        model = YOLO(os.path.join(save_root, model_name,'weights','best.pt'))  # load an official model
        metrics = model.val(save_dir = os.path.join(save_root, save_name),cfg = "cfg_classify.yaml",data="/media/libinWorkSpace/YZK_FuBu/cls_dataset", cache=False)  # no arguments needed, dataset and settings remembered
        with open(os.path.join(save_root, model_name, "log_info.txt"), "w+") as f:
            log_info = "测试模型test_031217"
            f.write(log_info)
        f.close()
    elif mode == 'test':
        model_name = "train_BCELoss_0318_1357"
        save_name = "test_BCELoss_0318_1357"
        save_root = "classify"
        data_root = "/media/libinWorkSpace/YZK_FuBu/cls_dataset/val"
        model = YOLO(os.path.join(save_root, model_name,'weights','best.pt')).cuda()  # load an official model
        with open(os.path.join(save_root, model_name, "test_image_info.txt"), "w+") as f:
            for cls in os.listdir(data_root):  # 所有的数据种类
                cls_p = os.path.join(data_root, cls)
                for im_name in os.listdir(cls_p):
                    im_p = osp.join(cls_p, im_name)
                    img = Image.open(im_p)
                    res = model.predict(img, imgsz=640)
                    if res[0].probs.top1 != int(cls) or res[0].probs.top1conf.item() <= 0.9:
                        log_info = f"im_name:{im_name:<20}\tclass:{int(cls):<5}\t pred:{res[0].probs.top1:<5}\tprobs:{res[0].probs.top1conf.item()}\n"
                        f.write(log_info)
        f.close()
except Exception as e:
    print(e)

# val
# model = YOLO("/media/libinWorkSpace/ultralytics/classify/test_030711/weights/best.pt")
# model.val(cfg = "cfg_classify.yaml",data="/media/libinWorkSpace/YZK_FuBu/cls_dataset", cache=False)  # evaluate model performance on the validation set

