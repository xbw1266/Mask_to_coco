import cv2
import glob
from detectron2 import model_zoo
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
import time
import functools
import argparse
import os
import torch
import numpy as np

def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.time()
        value = func(*args, **kwargs)
        end_time = time.time()
        run_time = end_time - start_time
        print("Finished {} in {} secs".format(func.__name__, run_time))
        return value
    return wrapper_timer

class DetectronInf():
    def __init__(self, weights_dir):
        self.cpu_device = torch.device("cpu")
        if not os.path.exists(os.path.join(weights_dir, "model_final.pth")):
            raise FileExistsError("Not found weights!")
        self.cfg = get_cfg()
        
        # self.metadata = MetadataCatalog.get("__unused")
        register_coco_instances(f"baxter_train", {}, f"baxter/train.json", f"baxter/train")
        register_coco_instances(f"baxter_test", {}, f"baxter/test.json", f"baxter/test")
        self.cfg.DATASETS.TEST = ("baxter_test", )
        self.cfg.DATALOADER.NUM_WORKERS = 2
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        self.metadata = MetadataCatalog.get("__unused")
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = os.path.join(weights_dir, "model_final.pth")
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
        self.predictor = DefaultPredictor(self.cfg)
    
    @timer
    def test(self, img_path):
        img = cv2.imread(img_path)
        # img = img[200:316+690, 100:350+486]
        assert img is not None
        output = self.predictor(img)
        img_vis = img[:, :, ::-1]
        assert "instances" in output
        instances = output["instances"].to(self.cpu_device)
        scores = (instances.scores).tolist()
        if len(scores) > 0:
            idx = scores.index(max(scores))
            instances = instances[idx]
        mask = (instances.pred_masks).numpy()
        mask = mask.squeeze()
        img = np.zeros((mask.shape))
        img[mask == True] = 1
        cv2.imshow("mask", img)
        visualizer = Visualizer(img_vis, self.metadata, scale=0.8, instance_mode=ColorMode.IMAGE )
        vis_output = visualizer.draw_instance_predictions(instances)
        rtn = vis_output.get_image()[:, :, ::-1]
        # rtn = cv2.cvtColor(rtn, cv2.COLOR_RGB2BGR)
        return rtn


if __name__ == "__main__":
    detector = DetectronInf("./output")
    # img_path = glob.glob("../new_data/bag1080p/*.jpg")
    img_path = glob.glob("./swapped_bg_cropped/*.jpg")
    for img in img_path:
        cv_img = detector.test(img)
        cv2.imshow("aaa", cv_img)
        if cv2.waitKey(0) & 0xff == ord('q'):
            break