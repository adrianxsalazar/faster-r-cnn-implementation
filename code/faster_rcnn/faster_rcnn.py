#This code has been created by Adrian Salazar Gomez at the University of
#Lincoln.

#The code uses the detectron 2 framework made by facebookresearch whose repo is in:
#https://github.com/facebookresearch/detectron2

#The goal of this code is to relesease a ready to go semi-detailed faster-rcnn
# implementation. With this detectron 2 faster r cnn we can choose a stoping
# criteria based on the validation set, which is not implemented in a standard
#detectron 2 implementation.

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
import json
import torch, torchvision
from torch.nn.parallel import DistributedDataParallel
import argparse
import numpy as np
import json
import matplotlib.pyplot as plt

#import detectron2
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (MetadataCatalog,build_detection_test_loader,build_detection_train_loader)
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import (COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,)

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
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
logger = logging.getLogger("detectron2")

#functions
def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """

    output_folder = os.path.join(cfg.OUTPUT_DIR)

    evaluator_list = []

    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))

    return DatasetEvaluators(evaluator_list)

#function to evaluate performace of the current model
def do_test(cfg, model):
    results = OrderedDict()

    for dataset_name in cfg.DATASETS.TEST:

      losses_total = []

      data_loader = build_detection_test_loader(cfg, dataset_name, DatasetMapper(cfg,True))

      for iteration_test, data_test in enumerate(data_loader):
        loss_dict = model(data_test)

        losses = sum(loss for loss in loss_dict.values())

        assert torch.isfinite(losses).all(), loss_dict

        loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}

        #get the total loss
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        losses_total.append(losses_reduced)

      mean_loss = np.mean(losses_total)

      print ('mean loss validation %s' %mean_loss)

      evaluator = get_evaluator(cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name))

      results_i = inference_on_dataset(model, data_loader, evaluator)

      results[dataset_name] = results_i

    if len(results) == 1:
        results = list(results.values())[0]

    return results, mean_loss

#training function
def do_train(cfg, model, resume=False):

    #start the training
    model.train()

    #configuration of the model based on the cfg
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    #chechpoints configuration
    checkpointer = DetectionCheckpointer(model,cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler)

    #depending on whether we are using a checkpoint or not the initial iteration
    #would be different
    if resume == False:
        start_iter=1
    else:
        start_iter = (checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1)

    #Number of iterations
    max_iter = cfg.SOLVER.MAX_ITER

    #checkpoints configurations
    periodic_checkpointer = PeriodicCheckpointer(checkpointer,cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter)
    checkpointer_best= DetectionCheckpointer(model,cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler)
    periodic_checkpointer_best= PeriodicCheckpointer(checkpointer_best, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter)

    #writer:
    writers = ([CommonMetricPrinter(max_iter), JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),] if comm.is_main_process() else [])

    #create the dataloader that get information from cfg.training set
    data_loader = build_detection_train_loader(cfg)

    #information about the current situation in the training process
    logger.info("Starting training from iteration {}".format(start_iter))

    #start iteration process (epochs)
    if resume == True:
      print ('Obtaining best val from previous session')
      best_loss=np.loadtxt(cfg.OUTPUT_DIR+"/"+"best_validation_loss.txt")
      print ('Previous best total val loss is %s' %best_loss)

    else:
        best_loss=99999999999999999999999999999999999

    #the patiente list stores the validation losses during the training process
    patience_list=[]
    patience_list.append(best_loss)

    dataset_size=cfg.NUMBER_IMAGES_TRAINING
    print("training set size is %s" %dataset_size)
    iteration_batch_ratio=int(round(float(dataset_size/cfg.SOLVER.IMS_PER_BATCH)))
    print ("%s Minibatches are cosidered as an entire epoch" %iteration_batch_ratio)

    with EventStorage(start_iter) as storage:
        if resume == True:
          iteration=start_iter
        else:
          start_iter=1
          iteration=1

        minibatch=0

        for data, miniepoch in zip(data_loader, range(start_iter*iteration_batch_ratio, max_iter*iteration_batch_ratio)):

            minibatch= minibatch +1
            if minibatch == iteration_batch_ratio:
              minibatch=0
              iteration = iteration + 1


            storage.step()

            loss_dict = model(data)
            #print (loss_dict)
            #print ('SPACE')

            losses = sum(loss for loss in loss_dict.values())
            #print (losses)
            #print ('SPACE')

            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            #print ('SPACE')

            #get the total loss
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            if minibatch == 0:
                print ("Minibatch %s / %s" %(minibatch, iteration_batch_ratio))
                print ("iteration %s / %s" %(iteration, max_iter))
                print ('Total losses %s \n' %losses_reduced)
                print (loss_dict_reduced)

            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)

            scheduler.step()

            #Test the validation score of the model
            if (cfg.TEST.EVAL_PERIOD > 0 and iteration % cfg.TEST.EVAL_PERIOD == 0 and iteration != max_iter and minibatch ==0 ):

                results, loss_val =do_test(cfg, model)
                patience_list.append(loss_val)
                #Compared to "train_net.py", the test results are not dumped to EventStorage

                if loss_val < best_loss:
                  print ('saving best model')
                  best_loss=loss_val
                  array_loss=np.array([best_loss])

                  #save best model
                  checkpointer_best.save('best_model')
                  np.savetxt(cfg.OUTPUT_DIR+"/"+"best_validation_loss.txt", array_loss, delimiter=',')


                if len(patience_list) > cfg.patience + cfg.warm_up_patience:
                  print('Chenking val losses .......')

                  #Item obtained (patience) iterations ago
                  item_patience=patience_list[-cfg.patience]
                  continue_training=False

                  #Check whether the val loss has improved
                  for i in range(cfg.patience):
                    item_to_check=patience_list[-i]
                    if item_to_check < item_patience:
                      continue_training=True

                  if continue_training == True:
                    print ('The val loss has improved')

                  else:
                    print ('The val loss has not improved. Stopping training')
                    #print the validation losses
                    print (patience_list)

                    #Plot validation loss error evolution
                    plt.plot(range(1,len(patience_list)+1,1),patience_list)
                    plt.xlabel('iterations')
                    plt.ylabel('validation loss')
                    plt.title('Evolution validation loss: \n min val loss: '
                    +str(min(patience_list)))

                    #save the plot
                    plt.savefig(os.path.join(cfg.OUTPUT_DIR,'evolution_val_loss.png'))
                    break


                comm.synchronize()

            # if iteration - start_iter > cfg.TEST.EVAL_PERIOD and (iteration % cfg.TEST.EVAL_PERIOD == 0 or iteration == max_iter):
            #   for writer in writers:
            #     writer.write()

            if minibatch == 1:
              periodic_checkpointer.step(iteration)

def main(model_path, check_model, dataset, dataset_output, learning_rate,
images_per_batch, anchor_size, aspect_ratios, roi_thresh, number_classes,
patience,warm_up_patience,evaluation_period,standard_anchors):


    with open("./datasets/"+str(dataset)+"/json_train_set.json", 'rt', encoding='UTF-8') as annotations:
      coco = json.load(annotations)
    images = coco['images']
    annotations=coco['annotations']
    number_images=len(images)
    number_annotations=len(annotations)

    #Print statements
    print('Number of annotations: '+str(number_annotations))

    #training set
    register_coco_instances("my_dataset_train", {},
    "./datasets/"+str(dataset)+"/json_train_set.json","./datasets/"+str(dataset)+"/all/")

    #validation set
    register_coco_instances("my_dataset_val", {},
     "./datasets/"+str(dataset)+"/json_val_set.json", "./datasets/"+str(dataset)+"/all/")

    #testing set
    register_coco_instances("my_dataset_test", {},
    "./datasets/"+str(dataset)+"/json_test_set.json","./datasets/"+str(dataset)+"/all/")

    #iniciate the model
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_path)) #Get the basic model configuration from the model zoo

    #Check whether the model should load a checkpoint
    check=check_model
    if check == False:
        print ('loading standard model')
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_path)
    else:
        print ('loading trained model')
        pre_tained_model="./saved_models/faster_cnn/"+str(dataset_output)+"/best_model.pth"
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(pre_tained_model)


    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = roi_thresh  # set threshold for this model

    #training set
    cfg.DATASETS.TRAIN = ("my_dataset_train",)

    #Validation set
    cfg.DATASETS.TEST = ("my_dataset_val",)

    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = number_classes

    #TEST.EVAL_PERIOD controls every how many batches we evaluate the val loss
    cfg.TEST.EVAL_PERIOD = evaluation_period
    cfg.SOLVER.CHECKPOINT_PERIOD= evaluation_period
    cfg.SOLVER.IMS_PER_BATCH = images_per_batch
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512


    #Number of iterations. Adjust up if val mAP is still rising, adjust down if overfit
    cfg.SOLVER.MAX_ITER = 1000

    if standard_anchors == False:
        #Set up the anchor sizes
        list_aspect_anchor_size=args.anchor_size.split(',')
        list_aspect_anchor_size=[int(anchor) for anchor in list_aspect_anchor_size]
        cfg.MODEL.ANCHOR_GENERATOR.SIZES = [list_aspect_anchor_size]

        #set up the ratio sizes of the windows
        list_aspect_ratios=args.aspect_ratios.split(',')
        list_aspect_ratios=[float(ar) for ar in list_aspect_ratios]
        cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [list_aspect_ratios]

    #learning rate and decay
    cfg.SOLVER.BASE_LR = learning_rate
    cfg.SOLVER.GAMMA = 0.05

    #
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.NUMBER_IMAGES_TRAINING= number_images

    #Directory where we save the checkpoint and saved models
    cfg.OUTPUT_DIR="./saved_models/faster_cnn/"+str(dataset_output)+"/"
    #create the folder if the folder does not exist
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    #Set up patience pararameters. Warm up patience is the number of batches will
    # happen independently of whether the val improves. The patience is the
    # number of batches that the training will process after an improvement in
    # the validation loss. If the val loss does not improve in this period
    #training will stop.
    cfg.patience= patience
    cfg.warm_up_patience=warm_up_patience

    #create the model to train
    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))

    #start the training process
    do_train(cfg, model, resume=check)



if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description = 'introduce dataset folder')

    parser.add_argument('-model', metavar='model to load', required=False,
    default="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", type=str)

    parser.add_argument('-check_model', metavar='check if there is a checkpoint',
    required=False, default=False, type=bool)

    parser.add_argument('-model_output', metavar='where the model is gonna be stored',
    required=False, default="Dataset_A", type=str)

    parser.add_argument('-dataset', metavar='dataset to use', required=False,
    default='Dataset_A', type=str)

    parser.add_argument('-standard_anchors', metavar='standard_anchors', required=False,
    default=True, type=bool)

    parser.add_argument('-learning_rate', metavar='learning rate',
    required=False, default=0.0025, type=float)

    parser.add_argument('-images_per_batch', metavar='images_per_batch',
    required=False, default=6, type=int)

    parser.add_argument('-anchor_size', metavar='anchor_size',
    required=False, default='32,64,128,256,512', type=str)

    parser.add_argument('-aspect_ratios', metavar='aspect_ratios',
    required=False, default='0.5,1.0,2.0', type=str )

    parser.add_argument('-roi_thresh', metavar='roi thresh',
    required=False, default=0.5, type=float)

    parser.add_argument('-number_classes', metavar='number_classes',
    required=False, default=1, type=int)

    parser.add_argument('-patience', metavar='patience',
    required=False, default=20, type=int)

    parser.add_argument('-warm_up_patience', metavar='warm_up_patience',
    required=False, default=20, type=int)

    parser.add_argument('-evaluation_period', metavar='evaluation_period',
    required=False, default=5, type=int)

    args = parser.parse_args()

    main(args.model, args.check_model, args.dataset, args.model_output,
    args.learning_rate, args.images_per_batch, args.anchor_size, args.aspect_ratios,
    args.roi_thresh, args.number_classes, args.patience, args.warm_up_patience,
    args.evaluation_period, args.standard_anchors)
