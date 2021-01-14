#This code has been created by Adrian Salazar Gomez at the University of
#Lincoln.

#The code uses the detectron 2 framework made by facebookresearch whose repo is in:
#https://github.com/facebookresearch/detectron2

#The goal of this code is to relesease a ready to go semi-detailed faster-rcnn
# implementation. With this detectron 2 faster r cnn we can choose a stoping
# criteria based on the validation set, which is not implemented in a standard
#detectron 2 implementation.

#This testing reports the MAE, RMSE, and AP scores. It is optional to output
#the testing images with the infered bounding boxes.

#project structure
#./main
#-code/
#--faster_rcnn/
#---args_faster_rcnn.py
#-datasets
#--dataset A/
#---json_train_set.json
#---json_val_set.json
#---json_test_set.json
#---all/
#----images
#-saved_models/
#--faster_rcnn/
#---dataset A
#----best_model.pth
#--dataset A/

#If you have any question, please do not hesitate to contac me at my email:
#asalazargomez@lincoln.ac.uk

import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
import json
import argparse
import glob
import scipy
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error
import PIL.Image as Image
import random
import cv2

#import detectron2
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (MetadataCatalog,build_detection_test_loader,
build_detection_train_loader,build_detection_test_loader)

from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import (COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)

from detectron2 import model_zoo
from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog,
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode
from detectron2.engine import DefaultTrainer
logger = logging.getLogger("detectron2")

#Functions
def save_dictionary(dictpath_json, dictionary_data):
    a_file = open(dictpath_json, "w")
    json.dump(dictionary_data, a_file)
    a_file.close()

#main function for testing
def main(dataset, model, model_output, anchor_size, aspect_ratios, roi, supression,
 test_set,best_model,number_classes,save_infered_images):

    with open("./datasets/"+str(dataset)+"/json_test_set.json", 'rt', encoding='UTF-8') as annotations:
      coco = json.load(annotations)

    #info test set
    images = coco['images']
    annotations=coco['annotations']
    number_images_test=len(images)
    number_bb_test=len(annotations)
    print ('Total number of images in testing set:  %s' %number_images_test)
    print ('Total number of items to be counted is : %s' %number_bb_test)

    #training set
    register_coco_instances("my_dataset_train", {}, "./datasets/"+str(dataset)+"/json_train_set.json", "./datasets/"+str(dataset)+"/all/")

    #validation set
    register_coco_instances("my_dataset_val", {}, "./datasets/"+str(dataset)+"/json_val_set.json", "./datasets/"+str(dataset)+"/all/")

    #testing set
    register_coco_instances("my_dataset_test", {}, "./datasets/"+str(dataset)+"/"+str(test_set),"./datasets/"+str(dataset)+"/all/")
    cfg = get_cfg()

    #cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.MODEL.WEIGHTS= "./saved_models/faster_cnn/"+str(model_output)+"/"+str(best_model)

    #training set
    cfg.DATASETS.TRAIN = ("my_dataset_train",)

    #Validation set
    cfg.DATASETS.TEST = ("my_dataset_test",)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = roi

    #Non maximum supresion threshold parameter.
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = supression

    #directory where the results and best models are stored
    cfg.OUTPUT_DIR="./saved_models/faster_cnn/"+str(model_output)+"/"

    #Indicate the number of classes
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = number_classes

    #Choose the anchor sizes. The anchor size has to be the same as in the
    #training process.
    list_aspect_anchor_size=args.anchor_size.split(',')
    list_aspect_anchor_size=[int(anchor) for anchor in list_aspect_anchor_size]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [list_aspect_anchor_size]

    #Choose the ratios. The ratios has to be the same as in the training period.
    list_aspect_ratios=args.aspect_ratios.split(',')
    list_aspect_ratios=[float(ar) for ar in list_aspect_ratios]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [list_aspect_ratios]

    #Create model and start testing.
    predictor=DefaultPredictor(cfg)
    test_metadata = MetadataCatalog.get("my_dataset_test")

    #Lists and dictionaries to store the results
    pred= []
    gt = []
    dictionary_counts={}

    #Go through all the images
    for i in range(len(images)):

      image=images[i]
      image_id=image['id']
      image_name=image['file_name']

      #Get the ground data for the image.
      ground_truth=0
      for a in annotations:
        if a['image_id'] == image_id:
          ground_truth=ground_truth+1

      #Append the ground truth data
      gt.append(ground_truth)

      #read the image
      im=cv2.imread(os.path.join("./datasets/"+str(dataset)+"/all/",image_name))

      #Use our model to predict/infeer the bounding boxes
      outputs = predictor(im)

      #Count the number of boxes.
      prediction=(len(outputs['instances']))

      #Append the number of predicted bounding boxes to the list where we keep
      #track of the predictions.
      pred.append(prediction)
      dictionary_counts[str(image_name)]=round(abs(prediction),3)

      #Save the images with the infered bounding boxes
      if save_infered_images=True:
          cv2_imshow(out.get_image()[:, :, ::-1])

          #make directory to store the visual results
          os.makedirs(os.path.join(cfg.OUTPUT_DIR,'visual_results'), exist_ok=True)
          cv2.imwrite(os.path.join(cfg.OUTPUT_DIR,'visual_results',image_name), out.get_image()[:, :, ::-1])


    #Get the AP scores
    evaluator = COCOEvaluator("my_dataset_test", cfg, False, output_dir=
    os.path.join(cfg.OUTPUT_DIR,'results_folder'))
    test_loader = build_detection_test_loader(cfg,"my_dataset_test")
    ap_result = inference_on_dataset(predictor.model,test_loader,evaluator)


    #Calculate MAE and RMSE
    mae= mean_absolute_error(pred,gt)
    rmse = np.sqrt(mean_squared_error(pred,gt))
    count_results=np.array([mae,rmse])

    #create folder to store the results
    os.makedirs(os.path.join(cfg.OUTPUT_DIR,'results_folder'), exist_ok=True)
    #Dictionary with the number of infered items per image
    save_dictionary(os.path.join(cfg.OUTPUT_DIR,'results_folder',"dic_restults.json"),dictionary_counts)

    #Saving the counting and detection results.
    np.savetxt(os.path.join(cfg.OUTPUT_DIR,'results_folder',"counting_restults.txt"),count_results,delimiter=',')
    np.savetxt(os.path.join(cfg.OUTPUT_DIR,'results_folder',"detection_restults.txt"),ap_result,delimiter=',')

    return print ('testing done')

if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description = 'introduce dataset folder')

    parser.add_argument('-dataset', metavar='dataset to use', required=False,
    default='RiseholmeSingle130_strawberries', type=str)

    parser.add_argument('-model', metavar='model to load', required=False,
    default="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", type=str)

    parser.add_argument('-model_output', metavar='where the model is gonna be stored',
    required=False, default="RiseholmeSingle130_strawberries", type=str)

    parser.add_argument('-anchor_size', metavar='anchor_size',
    required=False, default='32,64,128,256,512', type=str)

    parser.add_argument('-aspect_ratios', metavar='aspect_ratios',
    required=False, default='0.5,1.0,2.0', type=str )

    parser.add_argument('-roi', metavar='roi',
    required=False, default=0.5, type=float)

    parser.add_argument('-supression', metavar='supression',
    required=False, default=0.5, type=float)

    parser.add_argument('-test_set', metavar='test_set',
    required=False, default='json_test_set.json', type=str)

    parser.add_argument('-best_model', metavar='best_model',
    required=False, default='best_model.pth', type=str)

    parser.add_argument('-number_classes', metavar='number of classes',
    required=False, default=1, type=int)

    parser.add_argument('-save_infered_images', metavar='save_infered_images',
    required=False, default=False, type=bool)

    args = parser.parse_args()

    main(args.dataset,args.model, args.model_output, args.anchor_size,
    args.aspect_ratios,args.roi, args.supression, args.test_set, args.best_model,
    args.number_classes, args.save_infered_images)
