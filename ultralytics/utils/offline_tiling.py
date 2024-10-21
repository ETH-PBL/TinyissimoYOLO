import torch
import cv2
import random
import yaml
import os

import numpy as np
import torchvision.ops as ops

from tqdm import tqdm
from dataclasses import dataclass
from typing import Generator, List, Tuple, Union

@dataclass
class Tile:
    x_min: int
    x_max: int
    y_min: int
    y_max: int

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return

        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1


class Tiler:

    def __init__(self, config):

        self.tile_dirs_exist = False
        self.read_config(config)
        self.model_input_size = self.config['network_input_size']
        self.target_nba = self.config['target_nba']
        self.dataset_yaml = self.config['dataset_yaml']
        self.mode = self.config['mode']
        self.keep_empty_tiles = self.config['keep_empty_tiles']
        self.filer_iou = self.config['filter_iou_threshold']
        self.filter_intersection_ratio = self.config['filter_intersection_ratio_threshold']
        self.read_dataset_yaml()
        self.dataset_dir = os.path.abspath(self.dataset_config['path'])
        self.create_new_dirs()

    def create_new_dirs(self):
        """
        Create new directories for the tiled images and labels.
        If the directories already exist, the function will not create them again.
        """
        self.new_train_dir = self.dataset_dir + '/train_tiles_target_nba_' + str(self.target_nba) +\
            '_input_size_' + str(self.model_input_size) + '_mode_' + self.mode
        self.new_val_dir = self.dataset_dir + '/val_tiles_target_nba_' + str(self.target_nba) +\
            '_input_size_' + str(self.model_input_size) + '_mode_' + self.mode
        self.new_test_dir = self.dataset_dir + '/test_tiles_target_nba_' + str(self.target_nba) +\
            '_input_size_' + str(self.model_input_size) + '_mode_' + self.mode
        
        if os.path.exists(self.new_train_dir):
            self.tile_dirs_exist = True
        else:    
            os.makedirs(self.new_train_dir + '/images')
            os.makedirs(self.new_train_dir + '/labels')
        if not os.path.exists(self.new_val_dir):
            os.makedirs(self.new_val_dir + '/images')
            os.makedirs(self.new_val_dir + '/labels')
        if os.path.exists(self.new_test_dir):
            self.tile_dirs_exist = True
        else:
            os.makedirs(self.new_test_dir + '/images')
            os.makedirs(self.new_test_dir + '/labels')

        # Update the dataset yaml
        self.dataset_config['train'] = self.new_train_dir.split('/')[-1] + '/images'
        self.dataset_config['val'] = self.new_val_dir.split('/')[-1] + '/images'
        self.dataset_config['test'] = self.new_test_dir.split('/')[-1] + '/images'

        with open(self.dataset_yaml, 'w') as f:
            yaml.dump(self.dataset_config, f)

    def get_split_dataset(self):
        """
        Split the dataset into train, val and test.
        If the dataset with current tile parameters already exists, the function
        will not split the dataset again.
        """
        if not self.tile_dirs_exist:
            self.load_dataset()
            self.get_tiled_splits()
        else:
            print("Tile directories already exist. If you want to re-tile the dataset, delete the existing tile directories.")

    def load_images(self, image_path: str) -> List[dict]:
        """
        Load the images from the dataset.
        
        :param image_path: Path to the images
        :return: images: List of dictionaries containing the images"""
        path = self.dataset_dir + '/' + image_path
        images = []
        for filename in tqdm(os.listdir(path)):
            img = cv2.imread(os.path.join(path, filename))
            images.append({'filename': filename, 'image': img})
        return images
    
    def load_labels(self, path: str) -> List:
        """
        Load the labels from the dataset.
        
        :param path: Path to the labels
        :return: labels: List of dictionaries containing the labels
        """
        labels_path = self.dataset_dir + '/' + path.split('/')[0] + '/labels'
        labels = []
        for filename in os.listdir(labels_path):
            with open(os.path.join(labels_path, filename), 'r') as f:
                instances = f.read().splitlines()
                instances = [instance.split(' ') for instance in instances]
                instances_float = []
                for instance in instances:
                    instances_float.append([float(coord) for coord in instance])

                labels.append({'filename': filename, 'instances': np.array(instances_float)})
        return labels

    def load_dataset(self) -> None:
        """"
        Load the images and labels from the dataset and sort them by filename.
        """
        self.og_train_images = self.load_images(self.dataset_config['original_images']['train'])
        self.og_train_images = sorted(self.og_train_images, key=lambda k: k['filename'])
        self.og_train_labels = self.load_labels(self.dataset_config['original_images']['train'])
        self.og_train_labels = sorted(self.og_train_labels, key=lambda k: k['filename'])

        self.og_val_images = self.load_images(self.dataset_config['original_images']['val'])
        self.og_val_images = sorted(self.og_val_images, key=lambda k: k['filename'])
        self.og_val_labels = self.load_labels(self.dataset_config['original_images']['val'])
        self.og_val_labels = sorted(self.og_val_labels, key=lambda k: k['filename'])

        self.og_test_images = self.load_images(self.dataset_config['original_images']['test'])
        self.og_test_images = sorted(self.og_test_images, key=lambda k: k['filename'])
        self.og_test_labels = self.load_labels(self.dataset_config['original_images']['test'])
        self.og_test_labels = sorted(self.og_test_labels, key=lambda k: k['filename'])

    def read_dataset_yaml(self) -> None:
        """
        Read the dataset yaml.
        
        :param dataset_yaml: Path to the dataset yaml
        """
        with open(self.dataset_yaml, 'r') as stream:
            try:
                self.dataset_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    def read_config(self, config: str) -> None:
        """
        Read the config yaml.
        
        :param config: Path to the config file
        """
        with open(config, 'r') as stream:
            try:
                self.config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    def get_avg_bbox_area(self, bboxes: List) -> float:
        """
        Compute the average normalized bounding box area in an image.
        
        :param bboxes: List of bounding boxes
        :return: Average normalized bounding box area
        """
        areas = []
        for bbox in bboxes:
            areas.append(bbox[3]* bbox[4])

        return np.mean(areas)
    
    def get_tile_wh(self, full_image_shape: Tuple, avg_nba: float) -> int:
        """
        Compute the optimal tile width/height to reach the target NBA.
        Tiles can't be larger than the original image but can't be smaller than the
        network input size.
        
        :param full_image_shape: Shape of the original image
        :param avg_nba: Average normalized bounding box area
        :return: tile_wh: Optimal tile width/height"""
        scale_factor = self.target_nba / avg_nba
        img_area = full_image_shape[0] * full_image_shape[1]
        tile_wh = np.ceil(np.sqrt(img_area / scale_factor)).astype(int)
        tile_wh = min(tile_wh, min(full_image_shape[0], full_image_shape[1]))
        return max(tile_wh, self.model_input_size)
    
    def get_list_of_tiles(self, og_image: np.array, avg_bbox_area: float,
                          debug_plot: bool=False) -> List[Tile]:
        """
        Get a list of tiles that cover the entire image such that the average
        normalized bounding box area in each tile is close to the target normalized
        bounding box area. If the mode is train, the requirement for tiles to have 
        a minimum overlap is skipped. This slightly reduces the number of tiles
        and allows for faster training.
        
        :param og_image: Original image
        :param avg_bbox_area: Average normalized bounding box area
        :param debug_plot: Whether to plot the tiles
        :return: tiles: List of Tile objects
        """

        if self.mode == 'test':
            optimal_tile_overlap = 1.5*np.sqrt(avg_bbox_area*og_image.shape[0]*og_image.shape[1])
        else:
            optimal_tile_overlap = 0
        tile_wh = self.get_tile_wh(og_image.shape, avg_bbox_area)
        w, h = og_image.shape[1], og_image.shape[0]

        num_tiles_w = np.ceil(w / tile_wh).astype(int)
        if num_tiles_w == 1:
            centers_x = np.array([w / 2])
        else:
            optimal_overlap_w = False
            while not optimal_overlap_w:
                centers_x = np.linspace(tile_wh / 2, w - tile_wh / 2, num=num_tiles_w)
                x_spacing = centers_x[1] - centers_x[0]
                overlap = tile_wh - x_spacing
                if overlap < optimal_tile_overlap:
                    num_tiles_w += 1
                else:
                    optimal_overlap_w = True

        num_tiles_h = np.ceil(h / tile_wh).astype(int)
        if num_tiles_h == 1:
            centers_y = np.array([h / 2])
        else:
            optimal_overlap_h = False
            while not optimal_overlap_h:
                centers_y = np.linspace(tile_wh / 2, h - tile_wh / 2, num=num_tiles_h)
                y_spacing = centers_y[1] - centers_y[0]
                overlap = tile_wh - y_spacing
                if overlap < optimal_tile_overlap:
                    num_tiles_h += 1
                else:
                    optimal_overlap_h = True

        tiles = []
        for x in centers_x:
            for y in centers_y:
                x_min = int(x - tile_wh / 2)
                x_max = int(x_min + tile_wh)
                y_min = int(y - tile_wh / 2)
                y_max = int(y_min + tile_wh)
                tile = Tile(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
                tiles.append(tile)

        if debug_plot:
            fake_image = og_image.copy()
            for tile in tiles:
                color = [random.randint(0, 255), random.randint(0, 255),
                            random.randint(0, 255)]
                cv2.rectangle(fake_image, (tile.x_min, tile.y_min),
                                (tile.x_max, tile.y_max), color, 2)
                cv2.imshow('fake_image', fake_image)
                cv2.waitKey(0)

        return tiles
    
    def extract_tile_from_image(self, image: np.array, tile: Tile) -> np.array:
        """
        Extracts a tile from the original image.
        
        :param image: Original image
        :param tile: Tile object
        :return: resized_tile: Resized tile
        """
        image_tile = image[tile.y_min:tile.y_max, tile.x_min:tile.x_max, :]
        resized_tile = cv2.resize(image_tile, (self.model_input_size, self.model_input_size))

        return resized_tile
    
    def check_bbox(self, x_min: int, y_min: int, w: int, h: int) -> bool:
        """
        Checks whether the bbox dimensions are valid in the sense that the width
        and height have to be larger than 0.

        :param x_min: minimal x value of bounding box
        :param y_min: minimal y value of bounding box
        :param w: width of bounding box
        :param h: height of bounding box
        :return: bbox_is_valid: is False when bbox dimensions are invalid
        """
        x_max = x_min + w
        y_max = y_min + h
        bbox_is_valid = True

        if x_max <= x_min:
            bbox_is_valid = False
            return bbox_is_valid
        if y_max <= y_min:
            bbox_is_valid = False
            return bbox_is_valid

        return bbox_is_valid

    def check_that_down_sampled_bbox_is_valid(self, x_min: int, y_min: int,
                                              w: int, h: int) -> Tuple[int, int, int, int]:
        """
        Checks whether the down sampled bbox is valid and returns the corrected bbox dimensions.
        
        :param x_min: minimal x value of bounding box
        :param y_min: minimal y value of bounding box
        :param w: width of bounding box
        :param h: height of bounding box
        :return: x_min, y_min, w, h: corrected bbox dimensions
        """
        if x_min + w > self.model_input_size:
            print("Invalid tile dimension!")
            w -= 1
        if y_min + h > self.model_input_size:
            print("Invalid tile dimension!")
            h -= 1

        return x_min, y_min, w, h

    def create_new_instance_inside_tile(self, old_instance: List, tile: Tile,
                                        img_shape: Tuple) -> Union[None, List]:
        """
        Determines which part of the old instance bbox is inside the tile and creates a
        new instance with a bbox that fits into the tile and has coordinates relative to
        the tile edges.

        :param old_instance: object instance dict in original image frame
        :param tile: tile object that overlaps with old_instance bbox
        :param img_shape: shape of the original image
        :return: new_instance: object instance dict in tile frame
        """

        bbox = old_instance[1:]
        x1 = (bbox[0] - bbox[2] / 2) * img_shape[1]
        y1 = (bbox[1] - bbox[3] / 2) * img_shape[0]
        w = bbox[2] * img_shape[1]
        h = bbox[3] * img_shape[0]

        width_scale_down = (tile.x_max - tile.x_min) / self.model_input_size
        height_scale_down = (tile.y_max - tile.y_min) / self.model_input_size

        if width_scale_down < 0 or height_scale_down < 0:
            print("Invalid tile selection! Tiles must be larger than network input size.")

        if x1 > tile.x_min:
            x_new = x1 - tile.x_min
            if x_new + w > tile.x_max - tile.x_min:
                new_width = tile.x_max - x_new - tile.x_min - 1
            else:
                new_width = w

        else:
            x_new = 0
            new_width = x1 + w - tile.x_min

        if y1 > tile.y_min:
            y_new = y1 - tile.y_min
            if y_new + h > tile.y_max - tile.y_min:
                new_height = tile.y_max - y_new - tile.y_min - 1
            else:
                new_height = h

        else:
            y_new = 0
            new_height = y1 + h - tile.y_min

        x_new_scaled = int(np.round(x_new / width_scale_down))
        y_new_scaled = int(np.round(y_new / height_scale_down))
        w_new_scaled = int(np.round(new_width / width_scale_down))
        h_new_scaled = int(np.round(new_height / height_scale_down))

        x_new_scaled, y_new_scaled, \
        w_new_scaled, h_new_scaled = self.check_that_down_sampled_bbox_is_valid(x_new_scaled,
                                                                                y_new_scaled,
                                                                                w_new_scaled,
                                                                                h_new_scaled)

        if self.check_bbox(x_new_scaled, y_new_scaled, w_new_scaled, h_new_scaled):
            new_instance = {
                            "label": 1,
                            "specific_label": old_instance[0],
                            "x": x_new_scaled,
                            "y": y_new_scaled,
                            "w": w_new_scaled,
                            "h": h_new_scaled
                        }
            # Convert back to relative coordinates
            x = (x_new_scaled + w_new_scaled / 2) / self.model_input_size
            y = (y_new_scaled + h_new_scaled / 2) / self.model_input_size
            w = w_new_scaled / self.model_input_size
            h = h_new_scaled / self.model_input_size
            new_instance = np.array([old_instance[0], x, y, w, h])
            return new_instance

        else:
            return None

    def check_whether_bbox_is_in_current_tile(self, instance: List, tile: Tile,
                                              img_shape: Tuple) -> Union[None, List]:
        """
        Checks whether the current bbox overlaps with the tile and if so returns 
        a new instance dict that is adapted to this tile in terms of size and coordinate frame.

        :param instance: dict containing info about class and bbox dimensions
        :param tile: instance of Tile class
        :return: None or new instance dict
        """
        bbox = instance[1:]
        x1 = (bbox[0] - bbox[2] / 2) * img_shape[1]
        x2 = (bbox[0] + bbox[2] / 2) * img_shape[1]
        y1 = (bbox[1] - bbox[3] / 2) * img_shape[0]
        y2 = (bbox[1] + bbox[3] / 2) * img_shape[0]
        if x1 >= tile.x_max or x2 <= tile.x_min or y1 >= tile.y_max or y2 <= tile.y_min:
            return None
        else:
            instance_inside_tile = self.create_new_instance_inside_tile(instance, tile, img_shape)
            return instance_inside_tile

    def write_tiled_images_to_disk(self, tiled_images: List, tiled_labels: List,
                                   new_dir: str) -> None:
        """
        Write the tiled images and labels (in YOLO format) to disk.
        
        :param tiled_images: List of dictionaries containing the tiled images
        :param tiled_labels: List of dictionaries containing the tiled labels
        :param new_dir: Directory to write the tiled images and labels to
        """
        idx = 0
        og_filenames = []
        for img, labels in tqdm(zip(tiled_images, tiled_labels)):
            if img['og_filename'] not in og_filenames:
                idx = 0
                og_filenames.append(img['og_filename'])
            yolo_annotations = ""

            for instance in labels['boundingBoxes']:
                cls = instance[0]
                x = instance[1]
                y = instance[2]
                w = instance[3]
                h = instance[4]
                yolo_annotations += (f'{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n')

            annotation_name = labels['image_name'].split('.')[0] + '_' + str(idx).zfill(4) + '.txt'
            annot_file_path = new_dir + '/labels/' + annotation_name
            with open(annot_file_path, 'w') as f:
                f.writelines(yolo_annotations)
            image_name = img['og_filename'].split('.')[0] + '_' + str(idx).zfill(4) + '.png'
            cv2.imwrite(new_dir + '/images/' + image_name, img['tiled_image'])
            idx += 1

    def compute_area_hist(self, labels: List, og: bool=False,
                          plot_hist: bool=False) -> np.array:
        """
        Compute the histogram of the normalized bounding box areas (NBA) in the images.
        
        :param labels: List of dictionaries containing the labels
        :param og: Whether to compute the histogram of the NBA in the original images
        :param plot_hist: Whether to plot the histogram of the NBA
        :return: hist: Histogram of the NBA
        """
        bbox_areas = []
        for image_labels in labels:
            if og:
                bbox_areas.append(self.get_avg_bbox_area(image_labels['instances']))
            else:
                bbox_areas.append(self.get_avg_bbox_area(image_labels['boundingBoxes']))

        hist, bin_edges = np.histogram(bbox_areas, bins = np.arange(0, 0.01, 0.0002))

        if plot_hist:
            import matplotlib.pyplot as plt
            plt.bar(bin_edges[:-1], hist, width = 0.0002)
            plt.xlim(min(bin_edges), max(bin_edges))
            plt.show()   

        return hist

    def get_tiled_splits(self, plot_hist: bool=False) -> None:
        """
        Split the images into tiles and write the tiled images and labels to disk.
        Save the tiles_dict to disk as well.
        Optinal: Plot the histogram of the NBA in the original and tiled images.
        
        :param plot_hist: Whether to plot the histogram of the NBA in the original and tiled images
        """

        print("Writing tiled images to disk...")
        
        # Train
        tiled_train_images, tiled_train_labels, \
            train_tiles_dict = self.split_images_into_tiles(self.og_train_images, self.og_train_labels)
        self.write_tiled_images_to_disk(tiled_train_images, tiled_train_labels, self.new_train_dir)
        with open(self.new_train_dir + '/tiles_dict.yaml', 'w') as f:
            yaml.dump(train_tiles_dict, f)

        # Val
        tiled_val_images, tiled_val_labels, \
            val_tiles_dict = self.split_images_into_tiles(self.og_val_images, self.og_val_labels)
        self.write_tiled_images_to_disk(tiled_val_images, tiled_val_labels, self.new_val_dir)
        with open(self.new_val_dir + '/tiles_dict.yaml', 'w') as f:
            yaml.dump(val_tiles_dict, f)

        # Test
        tiled_test_images, tiled_test_labels, \
            test_tiles_dict = self.split_images_into_tiles(self.og_test_images, self.og_test_labels)
        self.write_tiled_images_to_disk(tiled_test_images, tiled_test_labels, self.new_test_dir)
        with open(self.new_test_dir + '/tiles_dict.yaml', 'w') as f:
            yaml.dump(test_tiles_dict, f)

        if plot_hist:
            import matplotlib.pyplot as plt
            plt.rcParams.update({'font.size': 22})
            plt.rcParams["font.family"] = "Times New Roman"
            train_og_hist = self.compute_area_hist(self.og_train_labels, og=True)
            train_tiled_hist = self.compute_area_hist(tiled_train_labels)
            val_og_hist = self.compute_area_hist(self.og_val_labels, og=True)
            val_tiled_hist = self.compute_area_hist(tiled_val_labels)
            test_og_hist = self.compute_area_hist(self.og_test_labels, og=True)
            test_tiled_hist = self.compute_area_hist(tiled_test_labels)

            train_og_hist += val_og_hist
            train_tiled_hist += val_tiled_hist

            safron = np.array([238, 185, 2, 255])
            fandango = np.array([164, 3, 111, 255])

            fig = plt.figure()
            fig.set_size_inches(10, 15)
            ax1 = fig.add_subplot(211)
            ax1.bar(np.arange(0, 0.01, 0.0002)[:-1], train_og_hist, width = 0.0002, color=fandango/255)
            ax1.bar(np.arange(0, 0.01, 0.0002)[:-1], test_og_hist, width = 0.0002, color=safron/255)
            ax1.set_xlim(0, 0.01)
            ax1.set_title('Average NBA in Full Scale Images')
            ax1.legend(['Train', 'Test'])
            ax1.set_xlabel('Normalized Bounding Box Area (NBA)')
            ax1.set_ylabel('Number of Images')
            ax2 = fig.add_subplot(212)
            ax2.bar(np.arange(0, 0.01, 0.0002)[:-1], train_tiled_hist, width = 0.0002, color=fandango/255)
            ax2.bar(np.arange(0, 0.01, 0.0002)[:-1], test_tiled_hist, width = 0.0002, color=safron/255)
            ax2.set_title('Average NBA in Tiled Images with Target NBA = 0.006')
            ax2.legend(['Train', 'Test'])
            ax2.set_xlabel('Normalized Bounding Box Area (NBA)')
            ax2.set_ylabel('Number of Images')
            ax2.set_xlim(0, 0.01)
            plt.show()      
        
    def plot_tiled_images(self, image: np.array, tiled_images: List,
                          tiled_labels: List, tiles: List[Tile]) -> None:
        """
        Plot the original image, draw the tile boundaries and plot the 
        tiled images with the bounding boxes on top of them.

        :param image: Original image
        :param tiled_images: List of dictionaries containing the tiled images
        :param tiled_labels: List of dictionaries containing the tiled labels
        :param tiles: List of Tile objects
        """

        fake_image = image.copy()
        for img, lbls, tile in zip( tiled_images, tiled_labels, tiles):
            color = [223, 190, 166] # ETH Blau 40%
            cv2.rectangle(fake_image, (tile.x_min, tile.y_min),(tile.x_max, tile.y_max), color, 4)
            tile_img = img['tiled_image'].copy()
            print(img['og_filename'])

            for bbox in lbls['boundingBoxes']:
                x1 = int((bbox[1] - bbox[3] / 2) * self.model_input_size)
                y1 = int((bbox[2] - bbox[4] / 2) * self.model_input_size)
                x2 = int((bbox[1] + bbox[3] / 2) * self.model_input_size)
                y2 = int((bbox[2] + bbox[4] / 2) * self.model_input_size)
                cv2.rectangle(tile_img, (x1, y1), (x2, y2), color, 2)

            cv2.imshow('tile_image', tile_img)
            cv2.imshow('og_image', fake_image)
            cv2.waitKey(0)

    def convert_tiles_dict_to_yaml_compatible(self, tiles_dict: dict) -> dict:
        """
        Convert the tiles_dict to a dictionary that can be written to a yaml file.
        (Basically just convert the Tile objects to dictionaries)

        :param tiles_dict: Dictionary containing info about the tiles
        :return: tiles_dict_converted: Dictionary containing info about the
                                       tiles that can be written to a yaml file
        """
        tiles_dict_converted = {}
        for key, value in tiles_dict.items():
            tiles_dict_converted[key] = {'image_name': value['image_name'], 'tiles': []}
            for tile in value['tiles']:
                tiles_dict_converted[key]['tiles'].append({'x_min': tile.x_min,
                                                 'x_max': tile.x_max,
                                                 'y_min': tile.y_min,
                                                 'y_max': tile.y_max})
        return tiles_dict_converted

    def split_images_into_tiles(self, images: List, labels: List,
                                debug_plot: bool=False) -> Tuple[List, List, dict]:
        """
        Split the images into tiles and return the tiled images and labels as well
        as a dictionary containing info about the tiles such that they can be stitched
        back together later.

        :param images: List of dictionaries containing the images
        :param labels: List of dictionaries containing the labels
        :param debug_plot: Whether to plot the tiled images
        :return: tiled_images: List of dictionaries containing the tiled images
                 tiled_labels: List of dictionaries containing the tiled labels
                 tiles_dict: Dictionary containing info about the tiles 
        """

        tiled_images = []
        tiled_labels = []
        tiles_dict = {}
        num_empty_tiles = 0
        num_empty_tiles_that_are_used = 0
        idx = 0
        for image_dict, image_labels in zip(images, labels):
            image = image_dict['image']
            img_shape = image.shape
            avg_bbox_area = self.get_avg_bbox_area(image_labels['instances'])
            tiles_dict[idx] = {'image_name': image_labels['filename'],
                                'tiles': self.get_list_of_tiles(image, avg_bbox_area)}
            current_tiled_images = []
            current_tiled_labels = []
            
            for tile in tiles_dict[idx]['tiles']:
                bboxes_list = []
                number_of_bboxes_in_this_tile = 0
                for instance in image_labels['instances']:
                    new_instance = self.check_whether_bbox_is_in_current_tile(instance, tile, img_shape)
                    if new_instance is not None:
                        bboxes_list.append(new_instance)
                        number_of_bboxes_in_this_tile += 1
                if number_of_bboxes_in_this_tile == 0:
                    num_empty_tiles += 1

                # If the mode is test, we want to keep all the tiles
                # If the mode is train, we want to keep a fraction of the empty tiles such
                # that the network also learns to make (or not make) predictions on empty tiles
                empty_tile_frequency = 1
                if self.mode == 'train':
                    empty_tile_frequency = 16

                if number_of_bboxes_in_this_tile > 0 or (self.keep_empty_tiles and
                                                        num_empty_tiles % empty_tile_frequency == 0):
                    if number_of_bboxes_in_this_tile == 0:
                        num_empty_tiles_that_are_used += 1
                    tiled_image_dict = {
                        'image_name': image_labels['filename'],
                        'tile': {'x_min': tile.x_min,
                                'x_max': tile.x_max,
                                'y_min': tile.y_min,
                                'y_max': tile.y_max
                                },
                        'boundingBoxes': bboxes_list
                    }
                    resized_image = self.extract_tile_from_image(image, tile)
                    
                    current_tiled_images.append({'og_filename': image_dict['filename'], 'tiled_image': resized_image})
                    current_tiled_labels.append(tiled_image_dict)

            if debug_plot:
                print(f'AVG NBA: {avg_bbox_area}')
                self.plot_tiled_images(image, current_tiled_images, current_tiled_labels, tiles_dict[idx]['tiles'])

            tiled_images.extend(current_tiled_images)
            tiled_labels.extend(current_tiled_labels)

            idx += 1
        tiles_dict = self.convert_tiles_dict_to_yaml_compatible(tiles_dict)
        return tiled_images, tiled_labels, tiles_dict
    
    def get_intersection(self, bboxes1: torch.Tensor, bboxes2: torch.Tensor,
                         equal_boxes: bool = True) -> torch.Tensor:
        """
        Get the intersection area between two sets of bboxes.

        :param bboxes1: Tensor of shape (N, 4) containing the bounding boxes
        :param bboxes2: Tensor of shape (M, 4) containing the bounding boxes
        :param equal_boxes: Whether to calculate the intersection of a bbox with itself
        """
        bboxes1 = bboxes1.unsqueeze(1)
        bboxes2 = bboxes2.unsqueeze(0)

        x1 = torch.max(bboxes1[..., 0], bboxes2[..., 0])
        y1 = torch.max(bboxes1[..., 1], bboxes2[..., 1])
        x2 = torch.min(bboxes1[..., 2], bboxes2[..., 2])
        y2 = torch.min(bboxes1[..., 3], bboxes2[..., 3])

        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

        if equal_boxes:
            intersection.fill_diagonal_(0)

        return intersection
    
    def get_intersection_ratio(self, bboxes: torch.Tensor, intersection: torch.Tensor) -> torch.Tensor:
        """
        Get the intersection ratio between two sets of bboxes. The ratio is defined as
        the intersection area of bboxe A with bbox B divided by the area of bbox A.

        :param bboxes: Tensor of shape (N, 4) containing the bounding boxes
        :param intersection: Tensor of shape (N, N) containing the intersection areas
        :return: ratio: Tensor of shape (N, N) containing the intersection ratios
        """
        area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        ratio = intersection / area
        return ratio
    
    def get_all_bboxes_and_confs(self, stitched_pres: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get all bboxes and confidences from the stitched predictions.

        :param stitched_pres: Dictionary containing the stitched predictions
        :return: all_bboxes: Tensor of shape (N, 4) containing the bounding boxes
                 all_confs: Tensor of shape (N,) containing the confidences
        """
        all_bboxes = torch.empty(0, 4).to('cpu')
        all_confs = []
        for tile_idx in stitched_pres:
            tile = stitched_pres[tile_idx]
            bboxes = tile['predictions']
            for instance in bboxes:
                all_bboxes = torch.cat((all_bboxes, torch.tensor([instance['bbox']]).to('cpu')))
                conf = instance['conf'].to('cpu')
                all_confs.append(conf)

        all_confs = torch.tensor(np.asarray(all_confs)).to('cpu')

        return all_bboxes, all_confs
    
    def get_intersection_ratio_mask(self, bboxes: torch.Tensor) -> torch.Tensor:
        """
        Get a mask for the bboxes that have a high intersection ratio with another bbox.
        
        :param bboxes: Tensor of shape (N, 4) containing the bounding boxes
        :return: mask: Tensor of shape (N,) containing the mask
        """
        intersection = self.get_intersection(bboxes, bboxes)
        intersection_ratio = self.get_intersection_ratio(bboxes, intersection)
        max_ratio = torch.amax(intersection_ratio, dim=0)
        mask = torch.where(max_ratio < self.filter_intersection_ratio, True, False)

        return mask

    def non_max_suppression(self, stitched_preds: dict) -> Tuple[List, List]:
        """ For all bboxes check the iou with all other boxes and if the iou is
        higher than the threshold, remove the bbox that is smaller. 
        
        :param stitched_preds: Dictionary containing the stitched predictions
        :return: remaining_bboxes: List of remaining bounding boxes
                 remaining_confs: List of confidences for the remaining bboxes
        """

        all_bboxes, all_confs = self.get_all_bboxes_and_confs(stitched_preds)
        
        iou_mask = ops.nms(all_bboxes, all_confs, iou_threshold=self.filer_iou)
        remaining_bboxes = all_bboxes[iou_mask]
        remaining_confs = all_confs[iou_mask]

        ratio_mask = self.get_intersection_ratio_mask(remaining_bboxes)
        remaining_bboxes = remaining_bboxes[ratio_mask]
        remaining_confs = remaining_confs[ratio_mask]
            
        return remaining_bboxes, remaining_confs
    
    def get_correspondence_ids(self, intersection_ratio: torch.Tensor) -> List[List[int]]:
        """
        Get the correspondence ids for the bboxes that have a high intersection ratio with another bbox.

        :param intersection_ratio: Tensor of shape (N, N) containing the intersection ratios
        :param filter_intersection_ratio: Threshold for filtering intersection ratios
        :return: all_correspondence_ids: List of correspondence ids
        """
        all_correspondence_ids = []

        high_intersection_indices = (intersection_ratio > self.filter_intersection_ratio).nonzero()

        for i in range(intersection_ratio.shape[0]):
            correspondence_ids = [i]
            matching_indices = high_intersection_indices[high_intersection_indices[:, 0] == i, 1]
            correspondence_ids.extend(matching_indices.tolist())
            all_correspondence_ids.append(correspondence_ids)

        return all_correspondence_ids
    
    def merge_bboxes(self, stitched_preds: dict, plot: bool=False) -> Tuple[List, List]:
        """ 
        Merge overlapping bounding boxes that belong to the same object based on
        their intersection ratio. We use the intersection ratio to perform a one-way matching
        to find associations between bboxes. Associated bboxes are then merged into a single bbox.
        And we assign it the confidence of the bbox with the highest confidence score.

        :param stitched_preds: Dictionary containing the stitched predictions
        :param plot: Whether to plot the merged bboxes (useful for debugging)
        :return: merged_bboxes: List of merged bounding boxes
                 merged_confs: List of confidences for the merged bboxes

        """

        all_bboxes, all_confs = self.get_all_bboxes_and_confs(stitched_preds)
        intersection = self.get_intersection(all_bboxes, all_bboxes)
        intersection_ratio = self.get_intersection_ratio(all_bboxes, intersection)
        all_correspondence_ids = self.get_correspondence_ids(intersection_ratio)
        merged_sets = self.merge_sets(all_correspondence_ids)
        unique_correspondences = [sorted(list(s)) for s in merged_sets]

        if plot:
            fake_image = np.zeros((720, 1280, 3), dtype=np.uint8)
            for group_num, group in enumerate(unique_correspondences):
                print(f'Group {group_num}')
                color = [random.randint(0, 255), random.randint(0, 255),
                            random.randint(0, 255)]
                for i in group:
                    print(f'Instance {i}')
                    x1, y1, x2, y2 = all_bboxes[i]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(fake_image, (x1, y1), (x2, y2), color, 2)
                cv2.imshow('fake_image', fake_image)
                cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Merge the bboxes that are in the same group
        merged_bboxes = []
        merged_confs = []
        for correspondence_ids in unique_correspondences:
            bboxes = all_bboxes[correspondence_ids]
            confs = all_confs[correspondence_ids]
            x1 = torch.amin(bboxes[:, 0])
            y1 = torch.amin(bboxes[:, 1])
            x2 = torch.amax(bboxes[:, 2])
            y2 = torch.amax(bboxes[:, 3])
            merged_bboxes.append([x1, y1, x2, y2])
            merged_confs.append(torch.amax(confs))

        return merged_bboxes, merged_confs
    
    def merge_sets(self, list_of_sets: List[set]) -> List[set]:
        """
        Merge sets that share at least one value and return a list of sets containing only
        unique values.

        :param list_of_sets: List of sets
        :return: List of sets containing only unique values
        """
        elements = set().union(*list_of_sets)
        element_to_index = {element: i for i, element in enumerate(elements)}
        uf = UnionFind(len(elements))

        for s in list_of_sets:
            if len(s) > 1:
                it = iter(s)
                root = uf.find(element_to_index[next(it)])
                for element in it:
                    uf.union(root, uf.find(element_to_index[element]))

        sets_map = {}
        for i in range(len(elements)):
            root = uf.find(i)
            if root not in sets_map:
                sets_map[root] = set()
            sets_map[root].add(elements.pop())  # Pop elements from the set

        return list(sets_map.values())
    
    def stitch_tiled_predictions(self, tiled_predictions: Generator,
                                 tiles_dict: dict, image_name: str) -> Tuple[dict, List, List]:
        """
        Stitch the tiled predictions back together to the original image space.
        Overlapping predictions are merged based on their intersection ratio.

        :param tiled_predictions: Generator of tiled predictions
        :param tiles_dict: Dictionary containing the tile information
        :param image_name: Name of the image
        :return: stitched_predictions: Dictionary containing the stitched predictions
                 filtered_bboxes: List of filtered bounding boxes
                 filtered_confs: List of filtered confidences
        """
        
        for idx in tiles_dict:
            if tiles_dict[idx]['image_name'].split('.')[0] == image_name.split('.')[0]:
                tile_info = tiles_dict[idx]['tiles']
                tile_wh = tile_info[0]['x_max'] - tile_info[0]['x_min']
                break
        
        stitched_predictions = {}
        for tile_idx, (tile, pred) in enumerate(zip(tile_info, tiled_predictions)):
            stitched_predictions[tile_idx] = {'tile': {'x_min': tile['x_min'],
                                                     'x_max': tile['x_max'],
                                                     'y_min': tile['y_min'],
                                                     'y_max': tile['y_max']},
                                            'predictions': []}
            for box, conf in zip(pred.boxes.xyxy, pred.boxes.conf):
                x1, y1, x2, y2 = box
                x1 = x1 * tile_wh/self.model_input_size + tile['x_min']
                x2 = x2 * tile_wh/self.model_input_size + tile['x_min']
                y1 = y1 * tile_wh/self.model_input_size + tile['y_min']
                y2 = y2 * tile_wh/self.model_input_size + tile['y_min']
                stitched_predictions[tile_idx]['predictions'].append({'bbox': [x1, y1, x2, y2],
                                                                      'conf': conf})
        filtered_bboxes, filtered_confs = self.merge_bboxes(stitched_predictions)

        return stitched_predictions, filtered_bboxes, filtered_confs


if __name__ == "__main__":
    tiler = Tiler('/path/to/tiling_config.yaml')
    tiler.get_split_dataset()