#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import numpy as np

def plot_dist_transform(input_image, pixel_coords, dist_image, 
                        figsize=(8,8), plot_name='', add_title=False, 
                        colorbar=True,
                        poly_face_color='orange', poly_edge_color='red', 
                        poly_nofill_color='blue', cmap='bwr'):
    '''Explore distance transform'''

    fig, (ax0, ax1) = plt.subplots(1, 2,
                                        figsize=(2*figsize[0], figsize[1]))

    mind, maxd = np.round(np.min(dist_image),2), np.round(np.max(dist_image),2)
    
    if add_title:
        suptitle = fig.suptitle(plot_name.split('/')[-1], fontsize='large')

    # create patches
    patches = []
    patches_nofill = []
    if len(pixel_coords) > 0:
        # get patches    
        for coord in pixel_coords:
            patches_nofill.append(Polygon(coord, facecolor=poly_nofill_color, 
                                          edgecolor=poly_edge_color, lw=3))
            patches.append(Polygon(coord, edgecolor=poly_edge_color, fill=True, 
                                   facecolor=poly_face_color))
        p0 = PatchCollection(patches, alpha=0.25, match_original=True)
        p1 = PatchCollection(patches, alpha=0.75, match_original=True)

    ax0.imshow(input_image)
    ax0.axis('off')
    # if len(patches) > 0:
    #     ax0.add_collection(p0)
    # ax0.set_title('Normalized Input Image')
    ax0.title.set_fontsize(16)

    # transform
    # cbar_pointer = ax1.imshow(dist_image)
    # dist_suffix = " (min=" + str(mind) + ", max=" + str(maxd) + ")"
    # ax1.set_title("Binary Distance Transform" + dist_suffix)
    
    # overlay buildings on distance transform
    ax1.imshow(dist_image)
    # truth polygons
    # if len(patches) > 0:
    #     ax1.add_collection(p1)
    # truth mask
    #ax2.imshow(z, cmap=palette, alpha=0.5, 
    #       norm=matplotlib.colors.Normalize(vmin=0.5, vmax=0.9, clip=False))
    # ax1.set_title("Ground Truth Polygons Overlaid on Binary Distance Transform")
    ax1.title.set_fontsize(16)
    # if colorbar:
    #     #from mpl_toolkits.axes_grid1 import make_axes_locatable
    #     #divider = make_axes_locatable(ax2)
    #     #cax = divider.append_axes('right', size='5%', pad=0.05)
    #     #fig.colorbar(cbar_pointer, cax=cax, orientation='vertical')
    #     left, bottom, width, height = [0.38, 0.85, 0.24, 0.03]
    #     cax = fig.add_axes([left, bottom, width, height])
    #     # fig.colorbar(cbar_pointer, cax=cax, orientation='horizontal')

    plt.axis('off')
    plt.tight_layout()
    if add_title:
        suptitle.set_y(0.95)
        fig.subplots_adjust(top=0.96)
    plt.show()
 
    if len(plot_name) > 0:
        plt.savefig(plot_name)
    
    return
