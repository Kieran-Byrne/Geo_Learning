import pandas as pd
import cv2
from skimage import data, io
from PIL import Image


def x9_from_img(img):
    img_TL = img[:116,:485]
    img_TC = img[:116,485:970]
    img_TR = img[:116,970:1456]
    img_ML = img[116:232,:485]
    img_MC = img[116:232,485:970]
    img_MR = img[116:232,970:1456]
    img_BL = img[232:350,:485]
    img_BC = img[232:350,485:970]
    img_BR = img[232:350,970:1456]
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
