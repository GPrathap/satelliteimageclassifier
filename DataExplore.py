from __future__ import print_function

import glob
import json
import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np


from spaceNetUtilities import create_building_mask, create_dist_map, geojson_to_pixel_arr
from visualizer import plot_dist_transform, plot_truth_coords, plot_building_mask, plot_all_transforms


class DataExplore():

    def __init__(self, plugin_config):
        spacenet_util_dir = os.getcwd() + '/spaceNetUtilities'
        sys.path.extend([spacenet_util_dir])
        with open(plugin_config) as plugin_config:
            self.plugin_config = json.load(plugin_config)
            self.ground_truth_patches = []
            self.pos_val = 1
            self.pos_val_vis = 255
            self.number_of_images_to_read = int(self.plugin_config["number_of_images_to_read"])
            self.image_read_from = int(self.plugin_config["image_read_from"])
            self.spacenet_data_dir = str(self.plugin_config["data_dir"])
            self.spacenet_result_dir = str(self.plugin_config["result_dir"])
            self. pixel_coords_list = []
            self.init_folder_structure()
            self.load_images()



    def init_folder_structure(self):
        self.im_dir = self.spacenet_data_dir + '/3band/'
        self.vec_dir = self.spacenet_data_dir + '/vectordata/geojson/'
        self.im_dir_out = self.spacenet_result_dir + '/3band/'
        self.coords_demo_dir = self.spacenet_result_dir + '/pixel_coords_demo/'
        self.mask_dir = self.spacenet_result_dir + '/building_mask/'
        self.mask_dir_vis = self.spacenet_result_dir + '/building_mask_vis/'
        self.mask_demo_dir = self.spacenet_result_dir + '/mask_demo/'
        self.dist_dir = self.spacenet_result_dir + '/distance_trans/'
        self.dist_demo_dir = self.spacenet_result_dir + '/distance_trans_demo/'
        self.all_demo_dir = self.spacenet_result_dir + '/all_demo/'
        for p in [self.im_dir_out, self.coords_demo_dir, self.mask_dir, self.mask_dir_vis, self.mask_demo_dir,
                  self.dist_demo_dir, self.dist_dir, self.dist_demo_dir, self.all_demo_dir]:
            if not os.path.exists(p):
                os.mkdir(p)

    def load_images(self):
        self.raster_list = sorted(glob.glob(os.path.join(self.im_dir, '*.tif'))) [self.image_read_from:
        self.image_read_from +self.number_of_images_to_read]
        for im_tmp in self.raster_list:
            shutil.copy(im_tmp, self.im_dir_out)

    def plot_result(self):
        for i, rasterSrc in enumerate(self.raster_list):
            input_image = plt.imread(rasterSrc)
            print (i, "rasterSrc:", rasterSrc)
            name_root0 = rasterSrc.split('/')[-1].split('.')[0]
            self.plot_mapping_of_image(name_root0, input_image, rasterSrc)
        print("\nExplore pixel coords list...")
        print ("pixel_coords_list[2][0]:", self.pixel_coords_list[2][0])

    def plot_mapping_of_image_given_by_name(self, name):
        raster_src = glob.glob(name)[0]
        input_image = plt.imread(raster_src)
        print("rasterSrc:", raster_src)
        name_root0 = raster_src.split('/')[-1].split('.')[0]
        self.plot_mapping_of_image(name_root0, input_image, raster_src)

    def plot_mapping_of_image(self, name_root0, input_image, rasterSrc):
        # remove 3band or 8band prefix
        name_root = name_root0[6:]
        vectorSrc = self.vec_dir + 'Geo_' + name_root + '.geojson'
        maskSrc = self.mask_dir + name_root0 + '.tif'
        # pixel coords and ground truth patches
        pixel_coords, latlon_coords = \
            geojson_to_pixel_arr.geojson_to_pixel_arr(rasterSrc, vectorSrc,
                                                      pixel_ints=True,
                                                      verbose=False)
        self.pixel_coords_list.append(pixel_coords)
        plot_name = self.coords_demo_dir + name_root + '.png'
        patch_collection, patch_coll_nofill = \
            plot_truth_coords.plot_truth_coords(input_image, pixel_coords,
                                                figsize=(8, 8), plot_name=plot_name,
                                                add_title=False)
        self.ground_truth_patches.append(patch_collection)
        # building mask
        outfile = self.mask_dir + name_root0 + '.tif'
        outfile_vis = self.mask_dir_vis + name_root0 + '.tif'
        # create mask from 0-1 and mask from 0-255 (for visual inspection)
        create_building_mask.create_building_mask(rasterSrc, vectorSrc,
                                                  npDistFileName=outfile,
                                                  burn_values=int(self.pos_val))
        create_building_mask.create_building_mask(rasterSrc, vectorSrc,
                                                  npDistFileName=outfile_vis,
                                                  burn_values=self.pos_val_vis)
        plot_name = self.mask_demo_dir + name_root + '.png'
        mask_image = plt.imread(outfile)
        plot_building_mask.plot_building_mask(input_image, pixel_coords,
                                              mask_image,
                                              figsize=(8, 8), plot_name=plot_name,
                                              add_title=False)
        outfile = self.dist_dir + name_root0 + '.npy'  # '.tif'
        create_dist_map.create_dist_map(rasterSrc, vectorSrc,
                                        npDistFileName=outfile,
                                        noDataValue=0, burn_values=self.pos_val,
                                        dist_mult=1, vmax_dist=64)

        plot_name = self.dist_demo_dir + name_root + '.png'
        mask_image = plt.imread(maskSrc)
        dist_image = np.load(outfile)
        plot_dist_transform.plot_dist_transform(input_image, pixel_coords,
                                                dist_image, figsize=(8, 8),
                                                plot_name=plot_name,
                                                add_title=False)
        plot_name = self.all_demo_dir + name_root + '_titles.png'
        mask_image = plt.imread(maskSrc)
        dist_image = np.load(outfile)
        plot_all_transforms.plot_all_transforms(input_image, pixel_coords,
                                                mask_image, dist_image,
                                                figsize=(8, 8), plot_name=plot_name,
                                                add_global_title=False,
                                                colorbar=False,
                                                add_titles=True,
                                                poly_face_color='orange', poly_edge_color='red',
                                                poly_nofill_color='blue', cmap='bwr')

    def execute(self):
        self.plot_result()


plugin_config = "./config/config.json"
data_explore_util = DataExplore(plugin_config)
data_explore_util.plot_mapping_of_image_given_by_name('./result/3band/3band_AOI_1_RIO_img1008.tif')
# data_explore_util.plot_mapping_of_image_given_by_name('/data/train/AOI_5_Khartoum_Train/RGB-PanSharpen/RGB-PanSharpen_AOI_5_Khartoum_img1.tif')
# data_explore_util.execute()