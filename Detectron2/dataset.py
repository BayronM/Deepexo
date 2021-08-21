#!/usr/bin/env python3


import detectron2
from detectron2.structures import BoxMode
import numpy as np
import os, json, cv2, random
from detectron2.utils.visualizer import Visualizer
import matplotlib.pyplot as plt
from detectron2.data import detection_utils





def create_segmentation(px,py,size):
    range_px = list(range(px,px+size))
    range_py = list(range(py,py+size))
    box = []

    for i in range(0,size-1):
        box.extend([range_px[i],py])
        box.extend([range_px[i],py+(size-1)])
        box.extend([range_py[i],px])
        box.extend([range_py[i],px+(size-1)])
    return box



def get_exo_dict(data,position):

    dataset_dicts = []
    for idx in range(0,len(data)):
        record = {}
        height,width = data[idx].shape[0],data[idx].shape[1]
        record["file"] = data[idx]
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        objs = []
        #print(data[idx].shape)
        for planet in position[idx]:
            poly = [planet[0]-4,planet[1]+3,planet[0]-4,planet[1]-4,planet[0]+3,planet[1]-4,planet[0]+3,planet[1]+3]
            box = [np.min(poly[::2])-2,np.min(poly[1::2])-2,np.max(poly[::2])+2,np.max(poly[1::2])+2]
            obj = {
                "bbox" : box,
                "bbox_mode" : BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id":0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts






#dataset_dicts = get_exo_dict(preprocessed_data,position)


def show_random_example(data,dataset_dicts,metadata,n_examples):
    for d in random.sample(dataset_dicts,n_examples):
        a = data[d["image_id"]]
        visualizer = Visualizer((a*255).astype(np.uint8),metadata,scale=10.0)
        out = visualizer.draw_dataset_dict(d)
        #for planet in d["annotations"]:
        #    out = visualizer.draw_polygon(planet["segmentation"],color='g')
            #out = visualizer.draw_box(planet["bbox"],edge_color='r')
        plt.imshow(out.get_image(),cmap='RdBu_r')
        plt.show()
