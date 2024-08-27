import cv2
import numpy as np
import pandas as pd
import os
import shutil
from PIL import Image
from skimage import io
from params import *
from sklearn.model_selection import train_test_split
from catboost import Pool, CatBoostClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score

def crop_img(img):
    """To crop one file from the dataset shape
    to the exploitable shape"""
    if img.shape == (662,1536,3):
        return img[100:450,80:]
    else:
        print('SOURCE FILE ', img, ' NOT IN SHAPE (662,1536,3)')

def x9_from_img(img):
    """ To convert an image into an RGB array of 9 zones"""
    img_TL = img[:116,:485]
    img_TC = img[:116,485:970]
    img_TR = img[:116,970:1486]
    img_ML = img[116:232,:485]
    img_MC = img[116:232,485:970]
    img_MR = img[116:232,970:1486]
    img_BL = img[232:350,:485]
    img_BC = img[232:350,485:970]
    img_BR = img[232:350,970:1486]
    avg_TL = img_TL.mean(axis=0).mean(axis=0)
    avg_TC = img_TC.mean(axis=0).mean(axis=0)
    avg_TR = img_TR.mean(axis=0).mean(axis=0)
    avg_ML = img_ML.mean(axis=0).mean(axis=0)
    avg_MC = img_MC.mean(axis=0).mean(axis=0)
    avg_MR = img_MR.mean(axis=0).mean(axis=0)
    avg_BL = img_BL.mean(axis=0).mean(axis=0)
    avg_BC = img_BC.mean(axis=0).mean(axis=0)
    avg_BR = img_BR.mean(axis=0).mean(axis=0)
    x_dict = {'avg_TL_R' : [avg_TL[0]/255],
              'avg_TL_G' : [avg_TL[1]/255],
              'avg_TL_B' : [avg_TL[2]/255],
              'avg_TC_R' : [avg_TC[0]/255],
              'avg_TC_G' : [avg_TC[1]/255],
              'avg_TC_B' : [avg_TC[2]/255],
              'avg_TR_R' : [avg_TR[0]/255],
              'avg_TR_G' : [avg_TR[1]/255],
              'avg_TR_B' : [avg_TR[2]/255],
              'avg_ML_R' : [avg_ML[0]/255],
              'avg_ML_G' : [avg_ML[1]/255],
              'avg_ML_B' : [avg_ML[2]/255],
              'avg_MC_R' : [avg_MC[0]/255],
              'avg_MC_G' : [avg_MC[1]/255],
              'avg_MC_B' : [avg_MC[2]/255],
              'avg_MR_R' : [avg_MR[0]/255],
              'avg_MR_G' : [avg_MR[1]/255],
              'avg_MR_B' : [avg_MR[2]/255],
              'avg_BL_R' : [avg_BL[0]/255],
              'avg_BL_G' : [avg_BL[1]/255],
              'avg_BL_B' : [avg_BL[2]/255],
              'avg_BC_R' : [avg_BC[0]/255],
              'avg_BC_G' : [avg_BC[1]/255],
              'avg_BC_B' : [avg_BC[2]/255],
              'avg_BR_R' : [avg_BR[0]/255],
              'avg_BR_G' : [avg_BR[1]/255],
              'avg_BR_B' : [avg_BR[2]/255]}
    X = pd.DataFrame.from_dict(x_dict)
    return X

def crop_dataset():
    """ To make a copy of the full dataset
    cropped at the exploitable format.
    Still organized by country only"""

    count = 0
    if not os.path.isdir(os.path.join(PROJECT_PATH,'data','cropped_dataset')):
        os.mkdir(os.path.join(PROJECT_PATH,'data','cropped_dataset'))
        print('Created cropped_dataset folder')

    for dir in os.listdir(os.path.join(PROJECT_PATH,'data','compressed_dataset')):
        if not os.path.isdir(os.path.join(PROJECT_PATH,'data','cropped_dataset',dir)):
            os.mkdir(os.path.join(PROJECT_PATH,'data','cropped_dataset',dir))
            print('Created ', dir, ' folder')
        for file in os.listdir(os.path.join(PROJECT_PATH,'data','compressed_dataset',dir)):
            cropped_file_name = 'cropped_' + file
            if not os.path.isfile(os.path.join(PROJECT_PATH,'data','cropped_dataset',dir,cropped_file_name)):
                img = io.imread(os.path.join(PROJECT_PATH,'data','compressed_dataset',dir,file))
                if img.shape == (662,1536,3):
                    img = crop_img(img)
                    io.imsave(os.path.join(PROJECT_PATH,'data','cropped_dataset',dir,cropped_file_name), img)
                    count += 1
                    print('cropped and saved ', file, ' in the ', dir, ' folder. N°', count)
            else:
                print(cropped_file_name, ' already exists in ', dir)


def city_countryside_classification():
    """ Sort all pictures by city or countryside in
    data/biome_data_by_biome_cropped/"""
    model = CatBoostClassifier()
    model.load_model(os.path.join(PROJECT_PATH,'modules','model','catboost_cropped.cbm'))
    data_dir = os.path.join(PROJECT_PATH,'data')
    full_dir =  os.path.join(data_dir,'cropped_dataset')
    biome_dir = os.path.join(data_dir,'biome_data_by_biome_cropped')
    countryside_dir = os.path.join(biome_dir,'countryside')
    city_dir = os.path.join(biome_dir,'city')

    if not os.path.isdir(biome_dir):
        os.mkdir(biome_dir)

    if os.path.isdir(countryside_dir):
        shutil.rmtree(countryside_dir)
    if os.path.isdir(city_dir):
        shutil.rmtree(city_dir)

    os.mkdir(countryside_dir)
    os.mkdir(city_dir)

    count = 0
    for dir in os.listdir(full_dir):
        print(dir)

        countryside_subdirectories = os.path.join(countryside_dir,dir)
        city_subdirectories = os.path.join(city_dir,dir)


        os.mkdir(countryside_subdirectories)
        os.mkdir(city_subdirectories)

        for file in os.listdir(os.path.join(full_dir,dir)):
            if file.startswith('cropped'):
                image = io.imread(os.path.join(full_dir,dir,file))
                X = x9_from_img(image)
                result = model.predict(X)
                if result[0] == 1:
                    shutil.copyfile(os.path.join(full_dir,dir,file), os.path.join(city_subdirectories,file))
                else:
                    shutil.copyfile(os.path.join(full_dir,dir,file), os.path.join(countryside_subdirectories,file))
                count += 1
                print(file, ' has been transferred with a score of ', result[0],'. N°',count)


if __name__ == '__main__':
    crop_dataset()
    city_countryside_classification()
