# faster-r-cnn-implementation
Faster R-CNN implementation

<p class="aligncenter">
<img src="https://github.com/adrianxsalazar/faster-r-cnn-implementation/blob/master/readme_images/detection_sample.png" alt="detection sample">
</p>




```
project
│   README.md    
│
└───code   
│   │   ...
│   └───faster_rcnn
│       │   faster_rcnn.py                        #this folder
│       │   testing_faster_rcnn.py
│       │   ...
|       └───tools
|           |   decision_anchors.py
|           |   plot_results_faster_rcnn.py
│   
└───datasets
|   │   ...
|   └───dataset_A
|       |   json_test_set.json
|       |   json_train_set.json
|       |   json_val_set.json
|       └───all
|           | img.png
|   
└───saved_models
|   |
|   └───faster_cnn
|       |
|       └───dataset_A
```


Following this directory structure, running this faster r-cnn would be as ea

```

$ python3 code/faster_rcnn/faster_rcnn.py -dataset "dataset A"

```
