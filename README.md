# faster-r-cnn-implementation
Faster R-CNN implementation.

This repository contains a Faster R-CNN implementation. This implementation uses the detectron2 framework. Although the detectron2 framework is relatively easy to use, this implementation simplifies some aspects that are not straightforward to implement using his framework. The main goal of this implementation is to facilitate the implementation of Faster R-CNN. So, users without much coding knowledge can easily customise this object detection tool. The rest of this repo explains how to use the implementation.

<p class="aligncenter">
<img src="https://github.com/adrianxsalazar/faster-r-cnn-implementation/blob/master/readme_images/detection_sample.png" alt="detection sample">
</p>


What can you do with this implementation?
<ul>
 <li>Train object detection models with your custom datasets.</li>
 <li>Set up the characteristic of the model with few commands. We cover elements such as the rpn anchor size, stopping criteria, base-model used for training, etc.</li>
 <li>Test your model with just a few commands.</li>
</ul>

Before explaining how to use this implementation, I should point to the detectron2 framework. Detectron2 is a fantastic tool for object detection and segmentation.  You can get more information about this framework in the official <a href="https://github.com/facebookresearch/detectron2">repository.</a> If you want to know more about faster R-CNN, I recommend to start with the original article: "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks".

<h3> Directory structure </h3>

First, you need to follow the given directory structure. You can always change the code if you do not want to follow this structure. I will explain later which lines of code you need to change.


```

project
│   README.md    
│
└───code                                          #Folder where we store the code.
│   │   ...
│   └───faster_rcnn
│       │   faster_rcnn.py                        #faster rcnn training code.
│       │   testing_faster_rcnn.py                #faster rcnn testing code.
|       └───tools
|           |   decision_anchors.py               #K-means approach to choose anchor sizes.
|           |   plot_results_faster_rcnn.py       #Plot the results of the trained models.
│   
└───datasets                                      #Folder where we save the datasets.
|   │   ...
|   └───dataset_A                                 #Dataset folder. Each dataset has to have a folder.
|       |   json_test_set.json                    #COCO JSON annotation file of the testing images.
|       |   json_train_set.json                   #COCO JSON annotation file of the training images.
|       |   json_val_set.json                     #COCO JSON annotation file of the validation images.
|       └───all                                   #Folder where we place the images.
|           | img_1.png
|           | img_2.png
|           | ...
|   
└───saved_models                                  #Folder where we save the models.
    |   ...
    └───faster_cnn                                
        |   ...
        └───dataset_A                             #Folder where we save the models we trained using dataset A.
            └───best_model.pth                    #Model we get from training.

```


First, you need to follow the given directory structure. You can always change the code if you do not want to follow this structure. I will explain later which lines of code you need to change.


<h3> Running the code </h3>
Once you have this structure, place the training, testing, and validation coco JSON files in the datasets/<name_of_your_dataset>/ directory. Then you only have to rename them as "json_train_set.json", "json_test_set.json", and "json_val_set.json". Then, copy all the dataset images under the directory datasets/<name_of_your_dataset>/all/. Now, everything is ready to train our Faster R-CNN.


To train the model,  we need to run the following command in our terminal in the project folder.

```

$ python3 code/faster_rcnn/faster_rcnn.py

```


This command will not work. We need to indicate which dataset we want to use and the folder's name to store the trained models. We can register these elements with the commands "-dataset" and "-model_output". If our dataset name is "dataset A" and the folder's name where we want to store the model is "dataset A output", the new command will be as follows.


```

$ python3 code/faster_rcnn/faster_rcnn.py -dataset "dataset A" -model_output "dataset A output"

```

Now, you should be able to train your model. Besides the "-dataset" and "-model_output" commands, there are multiple commands to customise your models. Some useful commands are "-model" which allows us to choose a feature extractor from the detectron2 model zoo and "-learning_rate" to select the learning rate.  The command "-number_classes" indicates to our model how many classes are in the dataset,  "-patience" shows how many iterations without improvement in the validation loss we allow, and "evaluation_period" which indicates how frequently we evaluate our models in the validation set.

The following command trains a Faster RCNN in the "dataset A". The learning rate is 0.0002 and a patience of 20. A patience of 20 means that if the model does not improve in 20 validations checks, the training will stop.

```

$ python3 code/faster_rcnn/faster_rcnn.py -dataset "dataset A" -model_output "dataset A output" -learning_rate 0.0002 -patience 20

```

<h3> Testing the model </h3>
Once we finish with training, we can evaluate our model. The python file "testing_faster_rcnn.py" contains the code to test our models.  To do so,  we only need to run this file with the corresponding commands "-model" and "-model_output". The testing uses the model located in the "model_output". We will need to use the "model_output" we indicated during the training process. The following command tests the models that we trained before.

```

$ python3 code/faster_rcnn/testing_faster_rcnn.py -dataset "dataset A" -model_output "dataset A output"

```


We also included an approach to choose the anchor size in the faster R-CNN region proposal network. The code clusters the bounding boxes sizes of the dataset to propose the anchor sizes. We can use the commands "-anchor_size" and "-aspect_ratios" to set the anchor size during training. I will explain more about his process in following updates of this page.

The testing process outputs two files "counting_results.txt" and "detection_results.txt", which will be available in the output folder. The

```

project
│    
└───saved_models                                  #Folder where we save the models.
    |   ...
    └───faster_cnn                                
        |   ...
        └───dataset_A                             #Folder where we save the models we trained using dataset A.
            └───results_folder
                |   counting_restults.txt         #Folder that contains the counting measures of our model such as MAE and RMSE
                |   detection_restults.txt        #Folder that contains the detection measures such as MaP


```

<h3> Training parameters </h3>

```

-> -model: description=standard model used for training,
            required=False, default="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", type=str

-> -check_model: description=check if there is a checkpoint from previous trainings processes,
            required=False, default=False, type=bool

-> -model_output: description=where the model is going be stored,
            required=False, default="dataset_A", type=str

-> -dataset: description=dataset to use,
            required=False, default="dataset_A", type=str

-> -standard_anchors: description=True if we want to use the standard anchor sizes, False if we want to suggest ours,
            required=False, default=True, type=bool

-> -learning_rate: description=learning rate in the training process
            required=False, default=0.0025, type=float

-> -images_per_batch: description=number of images used in each batch,
            required=False, default=6, type=int

-> -anchor_size: description= if -standard_anchors is True, the size of the anchors in the rpn,
            required=False, default='32,64,128,256,512', type=str

-> -aspect_ratios: description= if -standard_anchors is True, this indicates the aspect ration to use in the rpn
            required=False, default='0.5,1.0,2.0', type=str )

-> -roi_thresh: description=Overlap required between a ROI and ground-truth box in order for that ROI to be used as training example,
            required=False, default=0.5, type=float

-> -number_classes: description=number of classes,
            required=False, default=1, type=int

-> -evaluation_period: description= The command indicates the number of epochs required to evaluate our model in the validations set,
            required=False, default=5, type=int)

-> -patience: description= Number of evaluations without improvement required to stop the training process,
            required=False, default=20, type=int

-> -warm_up_patience: description=Number of evaluations that will happen independently of whether the validation loss improves,
            required=False, default=20, type=int


```


<h3> Choose the anchor size with k-means </h3>
We set up a command to choose the anchor size in the region proposal network. However, it might difficult to select the right size to improve your detection models. An approach to get the right anchor size is to cluster the bounding boxes of our dataset. Then we can find representative groups of bounding box sizes. We can use the representative centroids of the clustering process as the anchors in the rpm. The Yolo V2 paper proposed this method that led to improvements in the Yolo performance. Now, we can implement this in our faster R-CNNs.

To obtain the centroids, we need to run the following command. The parameter "-json_name" needs the name of the COCO JSON with the dataset labels. The "-dataset_path" command requires the path of the COCO JSON.  We can indicate the number of clusters with the attribute "-number_clusters".

```

$ python3 code/faster_rcnn/tools/decision_anchors.py -json_name "json_train_set.json" -dataset_path "./datasets/dataset A/" -number_cluster 9

```
