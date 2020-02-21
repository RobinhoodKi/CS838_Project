from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform, img_as_float, color, img_as_ubyte
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
# from IPython import display
# Ignore warnings
import warnings
import csv
import copy
from skimage import exposure
import random
from PIL import Image
import cv2

warnings.filterwarnings("ignore")

# plt.ion()   # interactive mode

class SegDataset(Dataset):

    def __init__(self, csv_file, transform=None):
     
        self.files_list = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        img = os.path.join(self.files_list.iloc[idx, 0])
        pmap = os.path.join(self.files_list.iloc[idx, 1])

        img = cv2.imread(img)
        pmap = cv2.imread(pmap, cv2.IMREAD_GRAYSCALE)

        sample = {'image':img, 'pmap':pmap}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
class Compose(object):


    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string



class ToTensor(object):
    def __call__(self, sample):
        img, pmap = sample['image'], sample['pmap']
        img = torch.from_numpy( np.array([img.transpose( (2, 0, 1) ).astype('float32')/255.]) )
#         pmap = torch.from_numpy( np.array([pmap.astype('float32')/255.]) )
        pmap = torch.from_numpy(np.array(pmap))
        return {'image': img.reshape(3, 512, 512), 'pmap': pmap.reshape(1, 512, 512)}
    
class Normalize(object):
#     def __init__(self, mean, std):
#          self.mean = mean
#          self.std = std

    def __call__(self, sample):
        
        #nparray
        input, output = sample['input'], sample['output']
        
#         gray = color.rgb2gray(input)
#         gray = exposure.equalize_hist(gray)
#         output = np.multiply(gray, output)
#         input[:, :, 0] = output
#         input[:, :, 1] = output
#         input[:, :, 2] = output
        output = exposure.adjust_gamma(output, 0.5)

        
        
        return {'input': input, 'output': output}

# default setting
def get_default_image_length():
    return 100
def get_default_input_channels():
    return 3
def get_default_batch_size():
    return 32
def get_default_num_workers():
    return 1

def generate_csv(path):
    """
    generate a csv file that has two columns of file 
    paths: one for images, the other for label paths

    file structure under path:

    |- image_file_pilot.csv
    |- images/
        |- image1.bmp
        |  ...  
    |- labels/
        |- label1.bmp
        |  ...
    
    """
    img_path = os.path.join(path, 'images')
    lab_path = os.path.join(path, 'labels')
    csv_file_path = os.path.join(path, 'image_files_pilot.csv')
    file_list_img = [name for name in os.listdir(img_path) if 
                      os.path.isfile(os.path.join(img_path, name))]
    file_list_img.sort()
    file_list_lab = [name for name in os.listdir(lab_path) if 
                      os.path.isfile(os.path.join(lab_path, name))]
    file_list_lab.sort()

    num_img = len(file_list_lab)
    print("Number of images: " + str(num_img))
    with open(csv_file_path, 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(0, num_img):
            
            img = os.path.join(img_path, str(file_list_img[i]))
            lab = os.path.join(lab_path, str(file_list_lab[i]))
            filewriter.writerow([img, lab])

def unnormalize_img(batch, mean, std):
    for img in batch:
        for t, m, s in zip(img, mean, std):
            t.mul_(s).add_(m)
    return batch 

def show_patch(dataloader, index = 3):
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(), 
              sample_batched['pmap'].size())

        # observe 4th batch and stop.
        if i_batch == index:
            plt.figure()
            input_batch, label_batch = sample_batched['image'], sample_batched['pmap']
            batch_size = len(input_batch)
            im_size = input_batch.size(2)
#             label_batch=label_batch.reshape([batch_size,1,im_size,im_size])
            print(label_batch.size())
#             input_batch = unnormalize_img(input_batch, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#             label_batch = unnormalize_img(label_batch, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

            grid = utils.make_grid(input_batch)
            plt.imshow(grid.numpy().transpose((1, 2, 0)))
            plt.figure()

            grid = utils.make_grid(label_batch)
            plt.imshow(grid.numpy().transpose((1, 2, 0)))
        
            plt.axis('off')
            plt.ioff()
            plt.show()
            break


def _match_cumulative_cdf(source, template):
    """
    Return modified source array so that the cumulative density function of
    its values matches the cumulative density function of the template.
    """
    src_values, src_unique_indices, src_counts = np.unique(source.ravel(),
                                                           return_inverse=True,
                                                           return_counts=True)
    tmpl_values, tmpl_counts = np.unique(template.ravel(), return_counts=True)

    # calculate normalized quantiles for each array
    src_quantiles = np.cumsum(src_counts) / source.size
    tmpl_quantiles = np.cumsum(tmpl_counts) / template.size

    interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)
    return interp_a_values[src_unique_indices].reshape(source.shape)



def match_histograms(image, reference, multichannel=False):
    """Adjust an image so that its cumulative histogram matches that of another.
    The adjustment is applied separately for each channel.
    Parameters
    ----------
    image : ndarray
        Input image. Can be gray-scale or in color.
    reference : ndarray
        Image to match histogram of. Must have the same number of channels as
        image.
    multichannel : bool, optional
        Apply the matching separately for each channel.
    Returns
    -------
    matched : ndarray
        Transformed input image.
    Raises
    ------
    ValueError
        Thrown when the number of channels in the input image and the reference
        differ.
    References
    ----------
    .. [1] http://paulbourke.net/miscellaneous/equalisation/
    """
    shape = image.shape
    image_dtype = image.dtype

    if image.ndim != reference.ndim:
        raise ValueError('Image and reference must have the same number of channels.')

    if multichannel:
        if image.shape[-1] != reference.shape[-1]:
            raise ValueError('Number of channels in the input image and reference '
                             'image must match!')

        matched = np.empty(image.shape, dtype=image.dtype)
        for channel in range(image.shape[-1]):
            matched_channel = _match_cumulative_cdf(image[..., channel], reference[..., channel])
            matched[..., channel] = matched_channel
    else:
        matched = _match_cumulative_cdf(image, reference)

    return matched           