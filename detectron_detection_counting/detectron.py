#import env

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import os,cv2

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode

from detectron2.data.datasets import register_coco_instances
register_coco_instances("watermelon", {}, "_annotations.coco.json", "train")
metadata = MetadataCatalog.get("watermelon")
dataset_dict=detectron2.data.datasets.load_coco_json("/content/drive/MyDrive/watermelon_counting/watermelon/val/_annotations.coco.json", "val", dataset_name="watermelon")
class_catalog = metadata.thing_classes

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("watermellon",)

import numpy
from PIL import Image
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 8000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1+1
# Focal loss
cfg.MODEL.RETINANET.FOCAL_LOSS_GAMMA = 2.0
cfg.MODEL.RETINANET.FOCAL_LOSS_ALPHA = 0.25
cfg.MODEL.RETINANET.SMOOTH_L1_LOSS_BETA = 0.1

cfg.MODEL.WEIGHTS = os.path.join("/content/drive/MyDrive/watermelon_counting/model/model_final.pth")  
# cfg.MODEL.DEVICE="cpu"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.DATASETS.TEST = ("watermellon", )
CLASSES = {1:'watermelon', 2:'background'}
CLASSES_OF_INTEREST = ("watermelon")

predictor = DefaultPredictor(cfg)

def convert_box_to_array(box):
    '''
    Detectron2 returns results as a Boxes class
    (https://detectron2.readthedocs.io/modules/structures.html#detectron2.structures.Boxes).
    These have a lot of useful extra methods (computing area etc) however can't be easily
    serialized, so it's a pain to transfer these predictions over the internet (e.g., if
    this is being used in the cloud, and wanting to send predictions back) so this method
    converts it into a standard array format. Detectron2 also returns results in (x1, y1, x2, y2)
    format, so this method converts it into (x1, y1, w, h).
    '''
    res = []
    for val in box:
        for ind_val in val:
            res.append(float(ind_val))

    # Convert x2, y2 into w, h
    res[2] = res[2] - res[0]
    res[3] = res[3] - res[1]
    return res

def get_bounding_boxes(image):
    '''
    Return a list of bounding boxes of objects detected,
    their classes and the confidences of the detections made.
    '''
   
    outputs = predictor(image)

    _classes = []
    _confidences = []
    _bounding_boxes = []

    for i, pred in enumerate(outputs["instances"].pred_classes):
        class_id = int(pred)
        _class = CLASSES[class_id]

        if _class in CLASSES_OF_INTEREST:
            _classes.append(_class)

            confidence = float(outputs['instances'].scores[i])
            _confidences.append(confidence)

            _box = outputs['instances'].pred_boxes[i]
            box_array = convert_box_to_array(_box)
            _bounding_boxes.append(box_array)
          
    #print(_bounding_boxes, _classes, _confidences)
    return _bounding_boxes, _classes, _confidences