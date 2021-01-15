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

Before explaining how to use this implementation, I should point to the detectron2 framework. Detectron2 is a fantastic tool for object detection and segmentation.  You can get more information about this framework in the official repository. If you want to know more about faster R-CNN, I recommend to start with the original article: "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks".


First, you need to follow the given directory structure. You can always change the code if you do not want to follow this structure. I will explain later which lines of code you need to change.


```
project
│   README.md    
│
└───code   
│   │   ...
│   └───faster_rcnn
│       │   faster_rcnn.py                        #faster rcnn training code
│       │   testing_faster_rcnn.py                #faster rcnn testing code
|       └───tools
|           |   decision_anchors.py               #K-means approach to choose anchor sizes
|           |   plot_results_faster_rcnn.py       #
│   
└───datasets
|   │   ...
|   └───dataset_A
|       |   json_test_set.json
|       |   json_train_set.json
|       |   json_val_set.json
|       └───all
|           | img_1.png
|           | img_2.png
|           | ...
|   
└───saved_models
|   |   ...
|   └───faster_cnn
|       |   ...
|       └───dataset_A
|       
```


Following this directory structure, running this faster r-cnn would be as ea

```

$ python3 code/faster_rcnn/faster_rcnn.py -dataset "dataset A"

```
