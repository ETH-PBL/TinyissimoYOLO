# import pdb, 
import os
import torch
import numpy as np
import cv2
import yaml
import argparse

import torchvision.ops as ops

from ultralytics import YOLO
from ultralytics.nn.tasks import get_size
from ultralytics.utils.offline_tiling import Tiler
from ultralytics import YOLO
from tqdm import tqdm


def load_tiled_test_images(dir):
    images = {}
    for filename in os.listdir(dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            full_image_name = filename[:-(len(filename.split('_')[-1])+1)]\
                + '.' + filename.split('.')[-1]
            if full_image_name not in images.keys():
                images[full_image_name] = []
            
            images[full_image_name].append(os.path.join(dir, filename))
    # Sort the list for each image
    for image in images.keys():
        images[image] = sorted(images[image])
    return images

def load_test_images(dir):
    images = {}
    for filename in os.listdir(dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):            
            # images.append(os.path.join(dir, filename.split('/')[-1]))
            images[filename] = [os.path.join(dir, filename)]
    # Sort the list for each image
    for image in images.keys():
        images[image] = sorted(images[image])
    return images

def compute_metrics(gt, pred_boxes, pred_conf, og_image, conf_thresh=0.6,
                    iou_thresh=0.5, plot=False):

    # convert pred to tensor
    pred = torch.tensor(pred_boxes)
    pred_conf = torch.tensor(pred_conf)
    pred = pred[pred_conf > conf_thresh]
    
    gt_boxes = torch.zeros((len(gt), 4))
    img_w, img_h = og_image.shape[1], og_image.shape[0]
    for i, instance in enumerate(gt):
        xc, yc, w, h = instance[1:5]
        xc, yc, w, h = xc * img_w, yc * img_h, w * img_w, h * img_h
        x1 = xc - w/2
        y1 = yc - h/2
        x2 = xc + w/2
        y2 = yc + h/2
        gt_boxes[i] = torch.tensor([x1, y1, x2, y2])
        # gt_boxes_list.append(gt_box)

    iou = ops.box_iou(gt_boxes, pred)
    gt_has_match = iou.amax(dim=1) > iou_thresh
    pred_has_match = iou.amax(dim=0) > iou_thresh

    tp = pred_has_match.sum()
    fp = (~pred_has_match).sum()
    fn = (~gt_has_match).sum()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    num_gt = len(gt)
    num_pred = len(pred)
    count_mae = abs(num_gt - num_pred)

    if plot:
        print(f"Count Mae: {count_mae}, F1: {f1:.2f}, Precision: {precision:.2f}, \
              Recall: {recall:.2f}, #GT: {num_gt}, #Pred: {num_pred}")
        filtered_image = og_image.copy()
        for i, instance in enumerate(gt_boxes):
            x1, y1, x2, y2 = instance
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if gt_has_match[i]:
                continue
            else:
                color = (255, 0, 0)
            cv2.rectangle(filtered_image, (x1, y1), (x2, y2), color, 3)
        for i, box in enumerate(pred):
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if pred_has_match[i]:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            cv2.rectangle(filtered_image, (x1, y1), (x2, y2), color, 3)
    
        cv2.imshow('filtered image', filtered_image)
        cv2.waitKey(0)

    return count_mae, precision, recall, f1

def plot_results(stitched_preds, filtered_boxes, filtered_conf, og_image, 
                 og_labels, conf_thresh=0.6):

    filtered_image = og_image.copy()
    for tile_idx in stitched_preds:
        tile = stitched_preds[tile_idx]['tile']
        instances = stitched_preds[tile_idx]['predictions']
        color = (np.random.randint(0, 255), np.random.randint(0, 255),
                 np.random.randint(0, 255))
        cv2.rectangle(og_image, (tile['x_min'], tile['y_min']),
                      (tile['x_max'], tile['y_max']), color, 2)

        for instance in instances:
            # if instance['conf'] > conf_thresh:
            x1, y1, x2, y2 = instance['bbox']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(og_image, (x1, y1), (x2, y2), color, 2)
        cv2.imshow('stitched image', og_image)
        cv2.waitKey(0)

    for instance in og_labels:
        xc, yc, w, h = instance[1:5]
        xc, yc, w, h = int(xc * og_image.shape[1]), int(yc * og_image.shape[0]),
        int(w * og_image.shape[1]), int(h * og_image.shape[0])
        x1 = xc - w//2
        y1 = yc - h//2
        x2 = xc + w//2
        y2 = yc + h//2
        cv2.rectangle(filtered_image, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.circle(filtered_image, (xc, yc), 5, (0, 0, 255), -1)
    for box, conf in zip(filtered_boxes, filtered_conf):
        if conf > conf_thresh:
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(filtered_image, (x1, y1), (x2, y2), (0, 255, 0), 4)
    
    cv2.imshow('filtered image', filtered_image)
    cv2.waitKey(0)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--use-tiling', default=True)
    parser.add_argument('--perform-iou-sweep', default=False)
    parser.add_argument('--plot', default=False)
    parser.add_argument('--image-set', default='test')
    parser.add_argument('--dataset-yaml-path', default='ultralytics/cfg/datasets/CARPK_tiling.yaml')


    args, unknown = parser.parse_known_args()

    image_set = args.image_set
    iou_sweep = args.perform_iou_sweep
    use_tiling = args.use_tiling
    dataset_yaml_path = args.dataset_yaml_path
    
    
    if use_tiling:
        tiler = Tiler('tiling_config.yaml')
        tiler.get_split_dataset()
        dataset_yaml = yaml.safe_load(open(dataset_yaml_path))
        data_dir = dataset_yaml['path'] + '/'
        test_images = load_tiled_test_images(data_dir + dataset_yaml[image_set])
        tiles_dict = yaml.safe_load(open(data_dir + dataset_yaml[image_set].replace('images', 'tiles_dict')
                                         + '.yaml'))
    else:
        dataset_yaml = yaml.safe_load(open(dataset_yaml_path))
        data_dir = dataset_yaml['path'] + '/'
        test_images = load_test_images(data_dir + dataset_yaml['original_images'][image_set])
    
    original_image_dir = data_dir + dataset_yaml['original_images'][image_set]
    original_labels_dir = data_dir + dataset_yaml['original_images'][image_set].replace('images', 'labels')

    model = YOLO('runs/detect/train782/weights/best.pt')
    

    full_count_mae = []
    full_precision = []
    full_recall = []
    full_f1 = []

    for image in tqdm(test_images, position=0, leave=True):
        # print(f"Processing {image}")
        og_image = cv2.imread(os.path.join(original_image_dir, image))
        og_labels_path = os.path.join(original_labels_dir, image.replace('png', 'txt'))
        with open(os.path.join(og_labels_path), 'r') as f:
            instances = f.read().splitlines()
            instances = [instance.split(' ') for instance in instances]
            instances_float = []
            for instance in instances:
                instances_float.append([float(coord) for coord in instance])

        gt_object_count = len(instances_float)
        result = model(test_images[image], stream=True)

        if use_tiling:
            stitched_preds, filtered_boxes, filtered_conf \
                = tiler.stitch_tiled_predictions(result, tiles_dict, image)
        else:
            filtered_boxes = []
            filtered_conf = []
            for pred in result:
                for box, conf in zip(pred.boxes.xyxy, pred.boxes.conf):
                    x1, y1, x2, y2 = box
                    filtered_boxes.append([x1, y1, x2, y2])
                    filtered_conf.append(conf)
        if iou_sweep:
            iou_thresh_vals = np.linspace(0.5, 0.95, 10)
            for iou_thresh in iou_thresh_vals:
                count_mae, pr, re, f1 = compute_metrics(instances_float, filtered_boxes,
                                                        filtered_conf, og_image,
                                                        iou_thresh=iou_thresh,
                                                        plot=args.plot)
                full_count_mae.append(count_mae)
                full_precision.append(pr)
                full_recall.append(re)
                full_f1.append(f1)
        else:
            count_mae, pr, re, f1 = compute_metrics(instances_float, filtered_boxes,
                                                    filtered_conf, og_image, plot=args.plot)
            full_count_mae.append(count_mae)
            full_precision.append(pr)
            full_recall.append(re)
            full_f1.append(f1)

    print(f"Average Count Mae: {np.nanmean(full_count_mae)},\
            Average Precision: {np.nanmean(full_precision)},\
            Average Recall: {np.mean(full_recall)},\
            Average F1: {np.nanmean(full_f1)}")