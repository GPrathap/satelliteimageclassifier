import math
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import skimage.morphology
import skimage.transform
import skimage.transform
import tables as tb
import cv2
# from dir_traversal_tfrecord import tfrecord_auto_traversal
import tqdm
from spaceNetUtilities import geojson_to_pixel_arr, create_dist_map, create_building_mask
from visualizer import plot_truth_coords, plot_building_mask, plot_dist_transform, plot_all_transforms


from DataProcessor import DataProcessor

plugin_config = "./config/config.json"
type_of_data="multi"

analysis_image_list = "./config/imageInfo.csv"

def prep_osmlayer_train(dataprocessor, image_id, input_image):
    area_id = dataprocessor.directory_name_to_area_id(dataprocessor.DATA_PATH)
    prefix = dataprocessor.area_id_to_prefix(area_id)
    dataprocessor.logger.info("prep_osmlayer for {}".format(prefix))

    layers = dataprocessor.extract_osmlayers(area_id)

    df = pd.read_csv(analysis_image_list, index_col='ImageId')

    df_sz = len(df)
    fn_tif = dataprocessor.get_train_image_path_from_imageid(image_id, dataprocessor.DATA_PATH, mul=False)
    with rasterio.open(fn_tif, 'r') as fr:
        values = fr.read(1).astype(np.float32)
        # fig, ax1 = plt.subplots(1, 2, sharey=True, figsize=(5, 5))

        fig1, ax1 = plt.subplots(1, 3, sharey=True, figsize=(6, 6))
        ax1[0].imshow(input_image)
        ax1[0].xaxis.set_visible(False)
        ax1[0].yaxis.set_visible(False)

        cou = 1
        for layer_geoms in layers:
            mask = rasterio.features.rasterize(
                layer_geoms,
                out_shape=values.shape,
                transform=rasterio.guard_transform(
                    fr.transform))
            mask_a = skimage.transform.resize(mask, (256, 256))
            imageInfo1 = np.array(mask_a)
            drawrgbonly1 = imageInfo1
            ax1[cou].imshow(drawrgbonly1)
            ax1[cou].xaxis.set_visible(False)
            ax1[cou].yaxis.set_visible(False)
            cou = cou + 1


def get_resized_8chan_image_train(dataprocessor, image_id, input_image):
    im = []
    area_id = dataprocessor.directory_name_to_area_id(dataprocessor.DATA_PATH)
    bs_rgb = dataprocessor.load_rgb_bandstats(area_id)
    bs_mul = dataprocessor.load_mul_bandstats(area_id)
    fn = dataprocessor.get_train_image_path_from_imageid(image_id, dataprocessor.DATA_PATH, mul=False)
    with rasterio.open(fn, 'r') as f:
        values = f.read().astype(np.float32)
        original = values
        original = np.swapaxes(original, 0, 2)
        original = np.swapaxes(original, 0, 1)
        fig1, ax1 = plt.subplots(1, 2, sharey=True, figsize=(5, 5))
        imageInfo = np.array(original)
        drawrgbonly = imageInfo
        ax1[0].imshow(drawrgbonly, aspect="auto")
        ax1[0].xaxis.set_visible(False)
        ax1[0].yaxis.set_visible(False)
        ax1[1].imshow(input_image, aspect="auto")
        ax1[1].xaxis.set_visible(False)
        ax1[1].yaxis.set_visible(False)


        for chan_i in range(3):
            min_val = bs_rgb[chan_i]['min']
            max_val = bs_rgb[chan_i]['max']
            values[chan_i] = np.clip(values[chan_i], min_val, max_val)
            values[chan_i] = (values[chan_i] - min_val) / (max_val - min_val)
            # if(chan_i==2):
            #     values = np.swapaxes(values, 0, 2)
            #     values = np.swapaxes(values, 0, 1)
            #     fig, ax = plt.subplots(1, 1, sharey=True, figsize=(5, 5))
            #     imageInfo = np.array(values)
            #     drawrgbonly = imageInfo
            #     ax.imshow(drawrgbonly, aspect="auto")
                
def get_slice_8chan_im(dataprocessor, image_id):
    area_id = dataprocessor.directory_name_to_area_id(dataprocessor.DATA_PATH)
    bandstats = dataprocessor.load_mul_bandstats(area_id)
    fn = dataprocessor.get_train_image_path_from_imageid(image_id, dataprocessor.DATA_PATH, mul=True)
    with rasterio.open(fn, 'r') as f:
        values = f.read().astype(np.float32)
        for chan_i in range(8):
            min_val = bandstats[chan_i]['min']
            max_val = bandstats[chan_i]['max']
            values[chan_i] = np.clip(values[chan_i], min_val, max_val)
            values[chan_i] = (values[chan_i] - min_val) / (max_val - min_val)
    values = np.swapaxes(values, 0, 2)
    values = np.swapaxes(values, 0, 1)
    assert values.shape == (650, 650, 8)

    fig, ax = plt.subplots(1, 1, sharey=True, figsize=(6, 6))
    imageInfo = np.array(values)
    drawrgbonly= imageInfo[:,:,[4,2,1]]
    ax.imshow(drawrgbonly, aspect="auto")
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    #
    fig, ax1 = plt.subplots(3, 3, sharey=True, figsize=(6, 6))
    indexx=-1
    for slice_pos in range(9):
        pos_j = int(math.floor(slice_pos / 3.0))
        pos_i = int(slice_pos % 3)
        if(pos_i==0):
            indexx=indexx+1
        x0 = dataprocessor.STRIDE_SZ * pos_i
        y0 = dataprocessor.STRIDE_SZ * pos_j
        im = values[x0:x0 + dataprocessor.INPUT_SIZE, y0:y0 + dataprocessor.INPUT_SIZE]
        assert im.shape == (256, 256, 8)

        imageInfo1 = np.array(im)
        drawrgbonly1 = imageInfo1[:, :, [4,2,1]]
        ax1[indexx, pos_i].imshow(drawrgbonly1, aspect="auto")
        ax1[indexx, pos_i].xaxis.set_visible(False)
        ax1[indexx, pos_i].yaxis.set_visible(False)



def _get_valtrain_mul_data(dataprocessor):
    area_id = dataprocessor.directory_name_to_area_id(dataprocessor.DATA_PATH)
    prefix = dataprocessor.area_id_to_prefix(area_id)
    fn_train = analysis_image_list
    df_train = pd.read_csv(fn_train)
    fn_im = dataprocessor.FMT_VALTRAIN_MUL_STORE.format(prefix)
    fn_mask = dataprocessor.FMT_VALTRAIN_MASK_STORE.format(prefix)
    with tb.open_file(fn_im, 'r') as f:
        with tb.open_file(fn_mask, 'r') as af:
            for idx, image_id in tqdm.tqdm(enumerate(df_train.ImageId.tolist())):
                pixel_coords_list = []
                ground_truth_patches = []
                spacenet_explore_dir = "/home/geesara/project/satelliteimageclassifier/result"
                coords_demo_dir = os.path.join(spacenet_explore_dir, 'pixel_coords_demo')

                maskDir = os.path.join(spacenet_explore_dir, 'building_mask')
                maskDir_vis = os.path.join(spacenet_explore_dir, 'building_mask_vis')
                mask_demo_dir = os.path.join(spacenet_explore_dir, 'mask_demo')

                distDir = os.path.join(spacenet_explore_dir, 'distance_trans')
                dist_demo_dir = os.path.join(spacenet_explore_dir, 'distance_trans_demo')

                all_demo_dir = os.path.join(spacenet_explore_dir, 'all_demo')
                pos_val, pos_val_vis = 1, 255
                im = np.array(f.get_node('/' + image_id))
                fig, ax = plt.subplots(1, 2, sharey=True, figsize=(6, 6))
                rgb_image = im[:, :, 0:3]
                input_image = rgb_image  # cv2.imread(rasterSrc, 1)
                name_root = image_id
                name_root0 = image_id
                rasterSrc =  os.path.join("/data/train/AOI_3_Paris_Train/RGB-PanSharpen/", 'RGB-PanSharpen_' + name_root + '.tif')
                # get name root

                # remove 3band or 8band prefix

                vectorSrc = os.path.join("/data/train/AOI_3_Paris_Train/geojson/buildings/", 'buildings_' + name_root + '.geojson')
                maskSrc = os.path.join(maskDir, name_root0 + '.tif')

                ####################################################
                # pixel coords and ground truth patches
                pixel_coords, latlon_coords = \
                    geojson_to_pixel_arr.geojson_to_pixel_arr(rasterSrc, vectorSrc,
                                                              pixel_ints=True,
                                                              verbose=False)
                pixel_coords_list.append(pixel_coords)

                plot_name = os.path.join(coords_demo_dir, name_root + '.png')
                patch_collection, patch_coll_nofill = \
                    plot_truth_coords.plot_truth_coords(input_image, pixel_coords,
                                                        figsize=(8, 8), plot_name=plot_name,
                                                        add_title=False)
                ground_truth_patches.append(patch_collection)
                plt.close('all')
                ####################################################

                ####################################################
                outfile = os.path.join(maskDir, name_root0 + '.tif')
                outfile_vis = os.path.join(maskDir_vis, name_root0 + '.tif')

                # create mask from 0-1 and mask from 0-255 (for visual inspection)
                create_building_mask.create_building_mask(rasterSrc, vectorSrc,
                                                          npDistFileName=outfile,
                                                          burn_values=pos_val)
                create_building_mask.create_building_mask(rasterSrc, vectorSrc,
                                                          npDistFileName=outfile_vis,
                                                          burn_values=pos_val_vis)

                plot_name = os.path.join(mask_demo_dir, name_root + '.png')
                mask_image = cv2.imread(outfile, 0)
                plot_building_mask.plot_building_mask(input_image, pixel_coords,
                                                      mask_image,
                                                      figsize=(8, 8), plot_name=plot_name,
                                                      add_title=False)
                plt.close('all')
                ####################################################

                ####################################################
                # signed distance transform
                # remove 3band or 8band prefix
                outfile = os.path.join(distDir, name_root0 + '.npy')  # '.tif'
                create_dist_map.create_dist_map(rasterSrc, vectorSrc,
                                                npDistFileName=outfile,
                                                noDataValue=0, burn_values=pos_val,
                                                dist_mult=1, vmax_dist=64)
                # plot
                # plot_name = os.path.join(dist_demo_dir + name_root, '_no_colorbar.png')
                plot_name = os.path.join(dist_demo_dir, name_root + '.png')
                #mask_image = plt.imread(maskSrc)  # cv2.imread(maskSrc, 0)
                dist_image = np.load(outfile)
                plot_dist_transform.plot_dist_transform(input_image, pixel_coords,
                                                        dist_image, figsize=(8, 8),
                                                        plot_name=plot_name,
                                                        add_title=False,
                                                        colorbar=True)  # False)
                plt.close('all')
                ####################################################

                ####################################################
                # plot all transforms
                plot_name = os.path.join(all_demo_dir, name_root + '.png')  # + '_titles.png'
                mask_image = plt.imread(maskSrc)  # cv2.imread(maskSrc, 0)
                dist_image = np.load(outfile)
                plot_all_transforms.plot_all_transforms(input_image, pixel_coords, mask_image, dist_image,
                                                        figsize=(8, 8), plot_name=plot_name, add_global_title=False,
                                                        colorbar=False,
                                                        add_titles=False,  # True,
                                                        poly_face_color='orange', poly_edge_color='red',
                                                        poly_nofill_color='blue', cmap='bwr')
                plt.close('all')
                ####################################################
                imageInfo = np.array(rgb_image)
                drawrgbonly= imageInfo[:,:,0:3]
                ax[0].imshow(drawrgbonly, aspect="auto")
                ax[0].xaxis.set_visible(False)
                ax[0].yaxis.set_visible(False)
                mask = np.array(af.get_node('/' + image_id))
                mask = (mask > 0.5).astype(np.uint8)

                # fig3, ax4 = plt.subplots(1, 1, sharey=True, figsize=(5, 5))
                ax[1].imshow(mask, aspect="auto")
                ax[1].xaxis.set_visible(False)
                ax[1].yaxis.set_visible(False)


                get_resized_8chan_image_train(dataprocessor, image_id, drawrgbonly)
                get_slice_8chan_im(dataprocessor, image_id)
                prep_osmlayer_train(dataprocessor, image_id, drawrgbonly)





dataprocessor = DataProcessor(plugin_config, base_model_name="v1", model_name="v1", image_dir="v1")
dataprocessor.execute()
_get_valtrain_mul_data(dataprocessor)


plt.show()



