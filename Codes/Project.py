import keras,os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np
import os.path
import cv2
import logging

from datetime import datetime
from collections import OrderedDict
import hdf5storage
from scipy import ndimage

import torch

from utils import utils_deblur
from utils import utils_logger
from utils import utils_model
from utils import utils_pnp as pnp
from utils import utils_sisr as sr
from utils import utils_image as util

datapath = r"C:\Users"
datapathcode = os.path.join(datapath,"Michel","Desktop","PROJET","Codes")
datapathtest = os.path.join(datapathcode, "testsets","testproject")
#importation du model
from keras.models import load_model
model = load_model(datapathcode+r"\checkpoint"+r"\vgg16_1.h5")

QualityList = []
for file in os.listdir(datapathtest) :
    img = image.load_img(os.path.join(datapathtest,file),target_size=(224,224))
    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)
    output = model.predict(img)
    imax = 0
    for i in range (1,len(output[0])):
        if output[0][i]>output[0][imax]:
            imax = i
    QualityList.append(imax)

noise_level_list = []
for i in range(len(QualityList)):
    if QualityList[i]==2:
        noise_level_list.append(0)
    if QualityList[i]==1:
        noise_level_list.append(35)
    if QualityList[i]==3:
        noise_level_list.append(60)
    if QualityList[i]==0:
        noise_level_list.append(100)

def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    noise_level_img = noise_level_list   # set AWGN noise level for noisy image
    noise_level_model = noise_level_img  # set noise level for model
    model_name = 'drunet_gray'           # set denoiser model, 'drunet_gray' | 'drunet_color'
    testset_name = 'testproject'               # set test set,  'bsd68' | 'cbsd68' | 'set12'
    x8 = False                           # default: False, x8 to boost performance
    show_img = False                     # default: False
    border = 0                           # shave boader to calculate PSNR and SSIM

    if 'color' in model_name:
        n_channels = 3                   # 3 for color image
    else:
        n_channels = 1                   # 1 for grayscale image

    model_pool = 'model_zoo'             # fixed
    testsets = 'testsets'                # fixed
    results = 'results'                  # fixed
    task_current = 'dn'                  # 'dn' for denoising
    result_name = testset_name + '_' + task_current + '_' + model_name

    model_path = os.path.join(datapathcode, model_pool, model_name+'.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    # ----------------------------------------
    # L_path, E_path, H_path
    # ----------------------------------------

    L_path = os.path.join(datapathcode, testsets, testset_name) # L_path, for Low-quality images
    E_path = os.path.join(datapathcode, results, result_name)   # E_path, for Estimated images
    util.mkdir(E_path)

    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(datapathcode, E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    # ----------------------------------------
    # load model
    # ----------------------------------------

    from models.network_unet import UNetRes as net
    model = net(in_nc=n_channels+1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    logger.info('Model path: {:s}'.format(model_path))
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info('Params number: {}'.format(number_parameters))

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []

    logger.info('model_name:{}, model sigma:{}, image sigma:{}'.format(model_name, noise_level_img, noise_level_model))
    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)

    for idx, img in enumerate(L_paths):

        # ------------------------------------
        # (1) img_L
        # ------------------------------------

        img_name, ext = os.path.splitext(os.path.basename(img))
        # logger.info('{:->4d}--> {:>10s}'.format(idx+1, img_name+ext))
        img_H = util.imread_uint(img, n_channels=n_channels)
        img_L = util.uint2single(img_H)

        # Add noise without clipping
        np.random.seed(seed=0)  # for reproducibility
        img_L += np.random.normal(0, noise_level_img[idx]/255., img_L.shape)

        util.imshow(util.single2uint(img_L), title='Noisy image with noise level {}'.format(noise_level_img[idx])) if show_img else None

        img_L = util.single2tensor4(img_L)
        img_L = torch.cat((img_L, torch.FloatTensor([noise_level_model[idx]/255.]).repeat(1, 1, img_L.shape[2], img_L.shape[3])), dim=1)
        img_L = img_L.to(device)

        # ------------------------------------
        # (2) img_E
        # ------------------------------------

        if not x8 and img_L.size(2)//8==0 and img_L.size(3)//8==0:
            img_E = model(img_L)
        elif not x8 and (img_L.size(2)//8!=0 or img_L.size(3)//8!=0):
            img_E = utils_model.test_mode(model, img_L, refield=64, mode=5)
        elif x8:
            img_E = utils_model.test_mode(model, img_L, mode=3)

        img_E = util.tensor2uint(img_E)

        # --------------------------------
        # PSNR and SSIM
        # --------------------------------

        if n_channels == 1:
            img_H = img_H.squeeze() 
        psnr = util.calculate_psnr(img_E, img_H, border=border)
        ssim = util.calculate_ssim(img_E, img_H, border=border)
        test_results['psnr'].append(psnr)
        test_results['ssim'].append(ssim)
        logger.info('{:s} - PSNR: {:.2f} dB; SSIM: {:.4f}.'.format(img_name+ext, psnr, ssim))

        # ------------------------------------
        # save results
        # ------------------------------------

        util.imsave(img_E, os.path.join(E_path, img_name+ext))

    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
    logger.info('Average PSNR/SSIM(RGB) - {} - PSNR: {:.2f} dB; SSIM: {:.4f}'.format(result_name, ave_psnr, ave_ssim))

if __name__ == '__main__':

    main()

