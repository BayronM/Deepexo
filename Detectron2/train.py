#!/usr/bin/env python3


import numpy as np
import os
from dataset import get_exo_dict, show_random_example
from data_preprocess import data_preprocess
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
import copy
import torch

#Dataset Configuration
TRAIN_IMAGES_DIR = "../data/CNN/master_training.npy"
PSF_IMAGES = "../data/CNN/tinyPSF.npy"
TEST_IMAGES_DIR = "../data/CNN/master_test.npy"
C_RATIO = [1,1000]
NAME_DATASET_TRAIN = "ExoDataset_train"
NAME_DATASET_TEST = "ExoDataset_test"


#Train Configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = (NAME_DATASET_TRAIN,)
cfg.DATASETS.TEST = (NAME_DATASET_TEST,)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)

#------------------------------------------------

def mapper(dataset_dict,data):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = data[dataset_dict["image_id"]]
    #image = np.tile(image,3)
    dataset_dict["image"] = torch.as_tensor(image)





def main():

    #Load the data
    print("-------------------\nTrain Data : ")
    data_train,_,_,position_train=data_preprocess(np.load(TRAIN_IMAGES_DIR),
                                            np.load(PSF_IMAGES),
                                            c_ratio=C_RATIO,
                                            no_blend = True)
    print("--------------------\nTest Data : ")
    data_test,_,_,position_test=data_preprocess(np.load(TEST_IMAGES_DIR),
                                           np.load(PSF_IMAGES),
                                           c_ratio=C_RATIO,
                                           no_blend = True)


    #Register the dataset for detectron2
    #Train dataset
    DatasetCatalog.register(NAME_DATASET_TRAIN, lambda a=data_train,b=position_train: get_exo_dict(a,b))
    MetadataCatalog.get(NAME_DATASET_TRAIN).set(thing_classes=["planet"])
    #Test dataset
    DatasetCatalog.register(NAME_DATASET_TEST, lambda a=data_test,b=position_test: get_exo_dict(a,b))
    MetadataCatalog.get(NAME_DATASET_TRAIN).set(thing_classes=["planet"])

    exo_metadata_train = MetadataCatalog.get(NAME_DATASET_TRAIN)
    exo_metadata_test = MetadataCatalog.get(NAME_DATASET_TEST)

    dataset_train_dict = get_exo_dict(data_train,position_train)
    show_random_example(data_train,dataset_train_dict,exo_metadata_train,1)


    data_loader = build_detection_train_loader(cfg,mapper=mapper)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()




if __name__ == "__main__":
    main()
