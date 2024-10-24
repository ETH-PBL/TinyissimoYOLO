import argparse
import os
import re
import cv2
import json
import urllib.request

import numpy as np


def load_gt_bbox(filepath):
    with open(filepath) as f:
        data = f.read()
    objs = re.findall(r'\d+ \d+ \d+ \d+ \d+', data)  # Use raw string
    annots = []
    for idx, obj in enumerate(objs):
        info = re.findall(r'\d+', obj)  # Use raw string
        x1 = float(info[0])
        y1 = float(info[1])
        x2 = float(info[2])
        y2 = float(info[3])
        width = x2 - x1
        height = y2 - y1
        x = x1 + 0.5 * width
        y = y1 + 0.5 * height
        instance = {"label": "car",
                    "coordinates": {
                        "x": x,
                        "y": y,
                        "width": int(width),
                        "height": int(height)
                    }}
        annots.append(instance)
    return annots


def plot_bboxes(image, instances):
    image_plot = np.copy(image)
    for instance in instances:
        width = instance["coordinates"]["width"]
        height = instance["coordinates"]["height"]
        x = int(instance["coordinates"]["x"] - 0.5 * width)
        y = int(instance["coordinates"]["y"] - 0.5 * height)
        start_point = (x, y)
        end_point = (x + width, y + height)
        color = (255, 0, 0)
        thickness = 2
        image_plot = cv2.rectangle(image_plot, start_point, end_point, color, thickness)

    cv2.imshow('annotated image', image_plot)
    cv2.waitKey(0)


def convert_carpk_to_create_ml(label_dir, images_dir, debug_plot=False):
    label_list = []
    for image_filename in os.listdir(images_dir):
        base_filename = (image_filename.strip().split('.'))[0]
        annot_filename = base_filename + '.txt'
        annotations = load_gt_bbox(label_dir + '/' + annot_filename)
        image_dict = {"image": image_filename,
                      "annotations": annotations,
                      "normalized_avg_bbox_area": -1,
                      "overlapping_bboxes_exist": True,
                      "top_down_view": True
                      }
        label_list.append(image_dict)

        if debug_plot and image_filename == "20160331_NTU_00066.png":
            img = cv2.imread(args.images_dir + '/' + image_filename)
            plot_bboxes(img, image_dict["annotations"])

    return label_list


def convert_create_ml_to_yolo(labels, image_dir, parent_dir):

    # Read the split files
    train_split_info = []
    for line in urllib.request.urlopen("https://github.com/mojulian/ultralytics/releases/download/0.1/train_images.txt"):
        train_split_info.append(line.decode('utf-8').split('.')[0])

    val_split_info = []
    for line in urllib.request.urlopen("https://github.com/mojulian/ultralytics/releases/download/0.1/val_images.txt"):
        val_split_info.append(line.decode('utf-8').split('.')[0])

    test_split_info = []
    for line in urllib.request.urlopen("https://github.com/mojulian/ultralytics/releases/download/0.1/test.txt"):
        test_split_info.append(line.decode('utf-8').split('\n')[0])

    train_folder = 'CARPK_train'
    val_folder = 'CARPK_val'
    test_folder = 'CARPK_test'

    for image in labels:
        image_name = image['image']
        image_name_wo_extension = image_name.split('.')[0]
        image_path = os.path.join(image_dir, image['image'])
        img = cv2.imread(image_path)
        img_res = img.shape[:2]

        yolo_annotations = ""

        for annot in image['annotations']:
            if annot['label'] == 'car':
                obj_class = 0
                x = annot['coordinates']['x']
                y = annot['coordinates']['y']
                width = annot['coordinates']['width']
                height = annot['coordinates']['height']

                x_center = x / img_res[1]
                y_center = y / img_res[0]
                w = width / img_res[1]
                h = height / img_res[0]

                # round to 6 decimal places
                x_center = round(x_center, 6)
                y_center = round(y_center, 6)
                w = round(w, 6)
                h = round(h, 6)

                # yolo_annotations.append([obj_class, x_center, y_center, w, h])
                yolo_annotations += (f'{obj_class} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n')
            else:
                print(f'Found an annotation with label {annot["label"]}. Skipping...')    

        # Create an annotation text file for the current image
        annot_file = image['image'].split('.')[0] + '.txt'
        label_folder = None
        if image_name_wo_extension in train_split_info:
            label_folder = train_folder + '/annotations'
            image_path = os.path.join(parent_dir, train_folder, 'images', image['image'])
            
        elif image_name_wo_extension in val_split_info:
            label_folder = val_folder + '/annotations'
            image_path = os.path.join(parent_dir, val_folder, 'images', image['image'])

        elif image_name_wo_extension in test_split_info:
            label_folder = test_folder + '/annotations'
            image_path = os.path.join(parent_dir, test_folder, 'images', image['image'])

        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        cv2.imwrite(image_path, img)

        annot_file_path = os.path.join(parent_dir, label_folder, annot_file)
        os.makedirs(os.path.dirname(annot_file_path), exist_ok=True)
        with open(annot_file_path, 'w') as f:
            f.writelines(yolo_annotations)
                       

        print(f'Created annotation file for {image["image"]}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', default='../../datasets/CARPK_devkit/data/Images', help='Path to the images directory in the CARPK dataset')
    parser.add_argument('--labels-dir', default='../../datasets/CARPK_devkit/data/Annotations', help='Path to the labels directory in the CARPK dataset')
    parser.add_argument('--new-data-dir', default='../../datasets/CARPK', help='Path to the new data directory in the YOLO format')

    args, unknown = parser.parse_known_args()
    create_ml_labels = convert_carpk_to_create_ml(args.labels_dir, args.images_dir, debug_plot=False)
    convert_create_ml_to_yolo(create_ml_labels, args.images_dir, args.new_data_dir)

    