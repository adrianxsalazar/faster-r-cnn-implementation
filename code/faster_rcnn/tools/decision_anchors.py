#This tool is to analyse the ground truth bounding boxes and recommend anchor
#sizes. Then.

#This approach were intially suggested in the YOLO paper:
#YOLO9000: Better, Faster, Stronger
#Link: https://arxiv.org/abs/1612.08242

#Part of this code was inpired by the code provided in:
#https://github.com/joydeepmedhi/Anchor-Boxes-with-KMeans

import numpy as np
import math
import argparse
import json
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


### functions ##################################################################
#function to calculate the intersection over the union between a box and an array
#of boxes
def iou(box, clusters):
    """
    box: numpy array with bounding boxes in width and height
    clusters: numpy array of shape (k, 2), where k is the number of clusters
    returns: numpy array of shape (k, 0), where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_

#function to transform a coco json file into a dataframe with the information we
#need such as the size of each bounding box, imag
def transform_coco_annotations_to_df(path_json_file):
    """
    path_json_file: path to the json file in coco format
    returns: dataframe of shape (r,10) with the characteristic we need,
    where r is the number of bounding boxes
    """

    #Open the json file
    with open(path_json_file, 'rt', encoding='UTF-8') as annotations:
        coco = json.load(annotations)

    #get the bounding boxes and images of the json file
    annotations = coco['annotations']
    images = coco['images']

    #Names of the keys we need from each annotation
    keys=['image_id','width','height','class_id','xmin','ymin','xmax','ymax',
    'width_box','height_box']

    #Dictionaries to map the image to its height and width
    height_dictionary={}
    for image in images:
        height_dictionary[image['id']]=image['height']

    width_dictionary={}
    for image in images:
        width_dictionary[image['id']]=image['width']

    #Array to store the information
    annotations_array=np.zeros((len(annotations),len(keys)))

    #populate the numpy array
    for annotation_index in range(len(annotations)):
        annotation=annotations[annotation_index]
        for i in range(len(keys)):
          if i == 0:
            annotations_array[annotation_index,i]=annotation['image_id']
          elif i == 1:
            annotations_array[annotation_index,i]=width_dictionary[annotation['image_id']]
          elif i == 2:
            annotations_array[annotation_index,i]=height_dictionary[annotation['image_id']]
          elif i == 3:
            annotations_array[annotation_index,i]=annotation['category_id']
          elif i == 4:
            annotations_array[annotation_index,i]=annotation['bbox'][0]
          elif i ==5:
            annotations_array[annotation_index,i]=annotation['bbox'][1]
          elif i ==6:
            annotations_array[annotation_index,i]=annotation['bbox'][0]+annotation['bbox'][2]
          elif i ==7:
            annotations_array[annotation_index,i]=annotation['bbox'][1]+annotation['bbox'][3]
          elif i ==8:
            annotations_array[annotation_index,i]=annotation['bbox'][2]
          else:
            annotations_array[annotation_index,i]=annotation['bbox'][3]


    #create a dataframe with the numpy array we created
    data= pd.DataFrame({keys[0]: annotations_array[:, 0],
                      keys[1]: annotations_array[:, 1],keys[2]: annotations_array[:, 2],
                      keys[3]: annotations_array[:, 3],keys[4]: annotations_array[:, 4],
                      keys[5]: annotations_array[:, 5],keys[6]: annotations_array[:, 6],
                      keys[7]: annotations_array[:, 7],keys[8]: annotations_array[:, 8],
                      keys[9]: annotations_array[:, 9]})

    #return the dataframe
    return data


#function to chage the format of the coco json annotations to weight and height.
def change_to_wh (data):
    """
    data: a dataframe with the bounding boxes size indicated with xmax, xmin ...
    returns: a dataframe with the bounding boxes width and height information
    """

    data['w'] = data['xmax'] - data['xmin']
    data['h'] = data['ymax'] - data['ymin']
    return data


# function from Tensorflow Object Detection API to resize image
def _compute_new_static_size(width, height, min_dimension, max_dimension):
    """
    width: width of the box
    height: height of the box
    min_dimension: min dimension of the images in the dataset
    max_dimension: max simension of the images in the dataset
    returns: resized bounding box width and height based on the dataset min and
    maximum width and height
    """

    orig_height = height
    orig_width = width
    orig_min_dim = min(orig_height, orig_width)

    # Calculates the larger of the possible sizes
    large_scale_factor = min_dimension / float(orig_min_dim)

    # Scaling orig_(height|width) by large_scale_factor will make the smaller
    # dimension equal to min_dimension, save for floating point rounding errors.
    # For reasonably-sized images, taking the nearest integer will reliably
    # eliminate this error.
    large_height = int(round(orig_height * large_scale_factor))
    large_width = int(round(orig_width * large_scale_factor))
    large_size = [large_height, large_width]
    if max_dimension:

        # Calculates the smaller of the possible sizes, use that if the larger
        # is too big.
        orig_max_dim = max(orig_height, orig_width)
        small_scale_factor = max_dimension / float(orig_max_dim)

        # Scaling orig_(height|width) by small_scale_factor will make the larger
        # dimension equal to max_dimension, save for floating point rounding
        # errors. For reasonably-sized images, taking the nearest integer will
        # reliably eliminate this error.
        small_height = int(round(orig_height * small_scale_factor))
        small_width = int(round(orig_width * small_scale_factor))
        small_size = [small_height, small_width]
        new_size = large_size

    if max(large_size) > max_dimension:
        new_size = small_size
    else:
        new_size = large_size

    return new_size[1], new_size[0]

#function to vinculate the centroid resulting from the clustering to standard
#squared anchor sizes with their aspect ratios. This is based on the iou similarity
def calculation_aspect_ratios(matrix_anchors,recommended_clusters):
    """
    matrix_anchors: array with shape (a,2), where a is the number of anchors
    we want to check for similiraty with the centroids.
    recommended_clusters: numpy array of shape (k, 0) where k is the number of
    clusters and returned centroids
    returns: dictionary where the keys are the each of the anchors we proposed
    in "matrix anchors" the values are a list of the aspect ratios recommended
    for that anchor to match one of the centroids.
    """

    #dictionary to store the results
    dict_results={}

    #create the keys of the dictionary
    for i in matrix_anchors:
        dict_results[i[0]]=[]

    #loop to go through all the centroids (recommended_clusters)
    for recommended_clusters_index in range(len(recommended_clusters)):

        box_to_match=[recommended_clusters[recommended_clusters_index][0],
        recommended_clusters[recommended_clusters_index][1]]

        #calulate the iou of the centroid with all the prposed anchors
        ious=iou(box_to_match,matrix_anchors)

        #Get the index of the max iou between the proposed anchors and centroids
        index_min=np.argmax(ious)
        max_similitude_anchor=matrix_anchors[index_min]
        array_to_match=np.array([[box_to_match[0],box_to_match[1]]])

        #check multiple aspect ratios for the most similar anchor. The goal
        #is to try several aspect ratios to improve the similarity
        scales=[0.01 + (x * 0.01) for x in range(0, 300)]
        max_iou=0
        max_scale=0
        for i in scales:
            scale=i
            area = max_similitude_anchor[0] ** 2.0
            w = math.sqrt(area/scale)
            h = scale * w
            scaled_anchor=[w,h]
            iou_scaled_anchor_with_array_to_match=iou(scaled_anchor,array_to_match)

            #we get the aspect ratio that return the maximum similarity
            if iou_scaled_anchor_with_array_to_match > max_iou:
                max_iou=iou_scaled_anchor_with_array_to_match
                max_scale=i

        #update the results dictionary
        d=[max_similitude_anchor[0]*(max_scale),max_similitude_anchor[1]/(max_scale)]
        list=dict_results[max_similitude_anchor[0]]
        list.append(round(max_scale,3))
        dict_results[max_similitude_anchor[0]]=list

    return dict_results

def calculation_aspect_ratios_refined(matrix_anchors,recommended_clusters,aspect_ratio_margin=0.03):
    """
    matrix_anchors: array with shape (a,2), where a is the number of anchors
    we want to check for similiraty with the centroids.
    recommended_clusters: numpy array of shape (k, 0) where k is the number of
    clusters and returned centroids.
    aspect_ratio_margin: float that indicates the similarity threshold between
    two aspect ratios in the same recommended anchor.
    returns: dictionary where the keys are the each of the anchors we proposed
    in "matrix anchors" the values are a list of the aspect ratios recommended
    for that anchor to match one of the centroids. The aspect ratios have to pass
    a similarity threshold
    """

    #dictionary to store the results
    dict_results={}
    list_scales=[]

    #create the keys of the dictionary
    for i in matrix_anchors:
        dict_results[i[0]]=[]

    #loop to go through all the centroids (recommended_clusters)
    for recommended_clusters_index in range(len(recommended_clusters)):

        box_to_match=[recommended_clusters[recommended_clusters_index][0],
        recommended_clusters[recommended_clusters_index][1]]

        #calulate the iou of the centroid with all the prposed anchors
        ious=iou(box_to_match,matrix_anchors)

        #Get the index of the max iou between the proposed anchors and centroids
        index_min=np.argmax(ious)
        max_similitude_anchor=matrix_anchors[index_min]
        array_to_match=np.array([[box_to_match[0],box_to_match[1]]])

        #check multiple aspect ratios for the most similar anchor. The goal
        #is to try several aspect ratios to improve the similarity
        scales=[0.01 + (x * 0.01) for x in range(0, 300)]

        max_iou=0
        max_scale=0

        for i in scales:
            scale=i
            area = max_similitude_anchor[0] ** 2.0
            w = math.sqrt(area/scale)
            h = scale * w
            scaled_anchor=[w,h]
            iou_scaled_anchor_with_array_to_match=iou(scaled_anchor,array_to_match)

            #we get the aspect ratio that return the maximum similarity
            if iou_scaled_anchor_with_array_to_match > max_iou:
                max_iou=iou_scaled_anchor_with_array_to_match
                max_scale=i

        d=[max_similitude_anchor[0]*(max_scale),max_similitude_anchor[1]/(max_scale)]

        #update the results dictionary considering whether the new suggestion is
        #to close to previous suggestions
        if len(list_scales) == 0:
            list_scales.append(max_scale)
            list=dict_results[max_similitude_anchor[0]]
            list.append(round(max_scale,3))
            dict_results[max_similitude_anchor[0]]=list

        between_indicator=False

        for scale in list_scales:
            if (scale-aspect_ratio_margin) <= max_scale <= (scale+aspect_ratio_margin):
                between_indicator=True

        if between_indicator == False:
            list_scales.append(max_scale)
            list=dict_results[max_similitude_anchor[0]]
            list.append(round(max_scale,3))
            dict_results[max_similitude_anchor[0]]=list

    return dict_results

#Function to calculate the average intersection over the union between multiple
#bounding boxes, which in our case is the ground truth boxes and the centroid boxes
def avg_iou(boxes, clusters):
    """
    boxes: numpy array of shape (r, 2), where r is the number of rows
    clusters: numpy array of shape (k, 2) where k is the number of clusters
    returns: average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])

#K means function that uses the iou as a distance measure
def kmeans(boxes, k, dist=np.median):
    """
    boxes: numpy array of shape (r, 2), where r is the number of rows
    k: number of clusters
    dist: distance function
    returns: numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed()

    # the Forgy method will fail if the whole array contains the same rows
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)

        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters

### end side functions #########################################################

def main(json_name,dataset_path,number_clusters):

    json_path=os.path.join(dataset_path,json_name)
    data=transform_coco_annotations_to_df(json_path)

    #Get the maximum and minimum dimensions
    min_dimension = min([data['width'].to_numpy()[1],data['height'].to_numpy()[1]])
    max_dimension = max([data['width'].to_numpy()[1],data['height'].to_numpy()[1]])

    #Resize the bounding boxes based on the information. This will not change
    #anything if all the images have the same size.
    data = change_to_wh(data)
    data['new_w'], data['new_h'] = np.vectorize(_compute_new_static_size)(data['width'],
                                                data['height'], min_dimension, max_dimension)

    #Data transformations
    data['b_w'] = data['new_w']*data['w']/data['width']
    data['b_h'] = data['new_h']*data['h']/data['height']

    #Scatter plot the width (b_w) and height (b_h) of the bounding boxes.
    sns.jointplot(x="b_w", y="b_h", data=data)

    #filename and filesave
    file_name='scatter_plot.png'
    saving_results_path=os.path.join(dataset_path,'clustering_bounding_boxes')
    os.makedirs(saving_results_path, exist_ok=True)

    plt.savefig(os.path.join(saving_results_path,file_name))
    plt.close()

    #set up data to cluster. Tramsform the data fram into a np array
    X = data[['b_w','b_h']].to_numpy()

    #list to store the average iou of the anchor centroids for each k
    average_iou=[]

    #Base 32 anchors. You can change this to a more detailed anchor size.
    matrix_anchors_extended=np.array([[32,32],[64.0,64],[96,96],[128,128],[160,160],
    [192,192],[224,224],[256,256],[288,288],[320,320],[352,352],[384,384],[416,416],
    [448,448],[480,480],[512,512]])

    #loop to
    for i in list(range(1,number_clusters+1)):
        cl = kmeans(X, i)
        avg=avg_iou(X,cl)

        file_name=str(i)+'_'+'centroids_for_k_.txt'
        np.savetxt(os.path.join(saving_results_path,file_name),cl)

        #We are linking the centroids we got with the base 32 centroids.
        #The function returns the most similar base 32 anchors and their respective
        #aspect ratios.
        centroids_association=calculation_aspect_ratios(matrix_anchors_extended,cl)

        file_name=str(i)+'_'+'centroids_association_for_k_.json'
        with open(os.path.join(saving_results_path,file_name), 'w') as json_file:
            json.dump(centroids_association, json_file)

        #Same as the previous centrid-square anchor association but in this case
        #if the linker anchor size and aspect ratios are too similar we  do not
        #consider it. This is just for simplicity.
        detailed_centroids_association=calculation_aspect_ratios_refined(matrix_anchors_extended,cl)
        file_name=str(i)+'_'+'centroids_detailed_association_for_k_.json'
        with open(os.path.join(saving_results_path,file_name), 'w') as json_file:
            json.dump(detailed_centroids_association, json_file)

        average_iou.append(avg)


    #Plot the evolution of the iou between the clusters and the bounding boxes
    #depending on the number of clusters
    plt.plot(list(range(1,number_clusters+1)),average_iou)
    plt.xticks(list(range(1,number_clusters+1)))
    plt.xlabel('Number of clusters')
    plt.ylabel('Average IoU')
    file_name='number_clusters_vs_average_iou.png'
    plt.savefig(os.path.join(saving_results_path,file_name))
    plt.close()

    return

if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description = 'introduce dataset folder')

    parser.add_argument('-json_name', metavar='json_name', required=True,
    type=str)

    parser.add_argument('-dataset_path', metavar='dataset_name', required=True,
    type=str)

    parser.add_argument('-number_clusters', metavar='number_clusters',
    required=False, default=8, type=int)

    args = parser.parse_args()

    main(args.json_name,args.dataset_path,args.number_clusters)
