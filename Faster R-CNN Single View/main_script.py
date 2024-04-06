# Set environment variables
import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'

import torch
print(torch.__version__, torch.cuda.is_available())

import sys, distutils.core
dist = distutils.core.run_setup("./detectron2/setup.py")
sys.path.insert(0, os.path.abspath('./detectron2'))

# Import some common libraries
import numpy as np
import cv2
import random
import pandas as pd
import io

# Import omidb
import omidb

# Import detectron2, detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger

# Setup detectron2 logger
setup_logger()

# Import some common detectron2 utilities
from detectron2.data import detection_utils as utils
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
cfg = get_cfg()
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode, Boxes, pairwise_iou

# Import our own trainer module and COCO evaluation.
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader, build_detection_train_loader

# Register Data
DatasetCatalog.clear()
def get_omidb_dicts(img_dir, csv_dir):
    df = pd.read_csv(csv_dir)
    
    dataset_dicts = []
    for idx, row in df.iterrows():
        record = {}
        filename = os.path.join(img_dir+"/HOLOGIC/ffdm/st"+"{0:03}".format(row["subtype"]), row["filename"])

        record["file_name"] = filename
        record["image_id"] = idx

        # Bounding box breast area         
        bbox = row["bbox"][12:-1]
        coords1 = bbox.split(',')
        r= np.array([0,0,0,0])
        indx1 = 0
        for c in coords1:
            aux = c.split('=')
            r[indx1]=(int(aux[1]))
            indx1 +=1

        # we can get width and heigth from bbox
        record["height"] = r[3]-r[1]
        record["width"] = r[2]-r[0]

        # Bounding box roi  
        bbox_roi = row["bbox_roi"][12:-1]
        coords2 = bbox_roi.split(',')
        s= np.array([0,0,0,0])
        indx2 = 0
        for c in coords2:
            aux = c.split('=')
            s[indx2]=(int(aux[1]))
            indx2 +=1
        bbox_roi = omidb.mark.BoundingBox(s[0]-r[0],s[1]-r[1],s[2]-r[0],s[3]-r[1])

        px = [bbox_roi.x1, bbox_roi.x2, bbox_roi.x2, bbox_roi.x1]
        py = [bbox_roi.y1, bbox_roi.y1, bbox_roi.y2, bbox_roi.y2]
        poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
        poly = [p for x in poly for p in x]
        objs = []
        obj =  {
                "bbox": [bbox_roi.x1 , bbox_roi.y1, bbox_roi.x2, bbox_roi.y2],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
                }
        objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

DatasetCatalog.register("omidb_train", lambda: get_omidb_dicts("/home/robert/data/mammo/iceberg_selection","/home/habtamu/Data_Split/training_set.csv"))
MetadataCatalog.get("omidb_train").set(thing_classes=["lesion"])

DatasetCatalog.register("omidb_val", lambda: get_omidb_dicts("/home/robert/data/mammo/iceberg_selection","/home/habtamu/Data_Split/validation_set.csv"))
MetadataCatalog.get("omidb_val").set(thing_classes=["lesion"])

#Training and val set dictionaries
train_metadata = MetadataCatalog.get("omidb_train")
train_dictionary = DatasetCatalog.get("omidb_train")

val_metadata = MetadataCatalog.get("omidb_val")
val_dictionary = DatasetCatalog.get("omidb_val")

# Custom trainer with data augmentation
class MyTrainerWithDataAugmentation(DefaultTrainer):

    @classmethod
    def build_train_loader(cls, cfg):
        augmentations = [
            T.RandomCrop("relative", (0.8, 1.0)),
            T.RandomBrightness(0.8, 1.8),
            T.RandomContrast(0.6, 1.3),
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False)
        ]
        mapper = DatasetMapper(cfg, is_train=True, augmentations=augmentations)
        return build_detection_train_loader(cfg, mapper=mapper)

# Custom trainer without data augmentation
class MyTrainer(DefaultTrainer):
    
    def __init__(self, cfg):
        super().__init__(cfg)

    def build_train_loader(self, cfg):
        return build_detection_train_loader(cfg)

# Setup the configuration for our dataset
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("omidb_train",)
cfg.DATASETS.TEST = ("omidb_val",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 34680
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
#trainer = MyTrainer(cfg)
trainer = MyTrainerWithDataAugmentation(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# Save config to file
config_file = 'config_final_34k_25_512_aug.yaml'
with open("output/"+config_file, "w") as f:
    f.write(cfg.dump())

# Test Evaluation through COCO Evaluator (Average Precision)
evaluator = COCOEvaluator("omidb_val", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "omidb_val")
print(inference_on_dataset(trainer.model, val_loader, evaluator))