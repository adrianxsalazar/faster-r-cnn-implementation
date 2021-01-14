import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import argparse
import os

def plot_results_faster_rcnn(results_path,dataset):
    list_nms=[0.5,0.6,0.7,0.8,0.9,0.95]
    mae_results=[]
    results_file_basename='_restults.txt'
    best_mae_value=1
    lowest_mae=99999999999

    for result_file_index in range(len(list_nms)):
        nms=list_nms[result_file_index]
        file_to_open=str(list_nms[result_file_index])+results_file_basename
        full_path_results_file=os.path.join(args.results_path,args.dataset,'results_folder',file_to_open)
        results_specific_nms=np.loadtxt(full_path_results_file)
        mae_results.append(results_specific_nms[0])

        if results_specific_nms[0] < lowest_mae:
            lowest_mae=results_specific_nms[0]
            best_mae_value=nms


    plt.plot(list_nms,mae_results)
    plt.xticks(list_nms)
    plt.xlabel('MAE')
    plt.ylabel('Non-maximum supression values')
    plt.title('MAE vs non-maximum supression values \n best mae value: ' +str(best_mae_value))
    plt.savefig(os.path.join(args.results_path,args.dataset,'results_folder','mae_results_for_all_nms.png'))



if __name__ == '__main__' :

    parser = argparse.ArgumentParser(description = 'introduce dataset folder')

    parser.add_argument('-results_path', metavar='results_path', required=False,
    default='./saved_models/faster_cnn/', type=str)

    parser.add_argument('-dataset', metavar='dataset', required=False,
    default="RiseholmeSingle130_strawberries", type=str)

    args = parser.parse_args()

    plot_results_faster_rcnn(args.results_path,args.dataset)
