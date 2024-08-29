import os
import cv2
import tensorflow as tf
import numpy as np

from tensorflow import keras
from keras import models
from keras.models import load_model
from PIL import Image
from modules.model.params import *
from modules.model.zoning import x9_from_img
from skimage import data, io
from catboost import CatBoostClassifier, Pool

def predict_city_countryside(img):
    model = CatBoostClassifier()
    model.load_model(CATBOOST_MODEL_PATH)

    X = x9_from_img(img)
    result = model.predict(X)
    return result[0] #if result[0] == 1 => city, if ==0 => countryside

def predict_country(img_resized, biome_predict):
    if biome_predict == 0:
        model = load_model(COUNTRYSIDE_MODEL_PATH)
        classes = ['Aland', 'Albania', 'American Samoa', 'Andorra', 'Antarctica', 'Argentina', 'Armenia', 'Australia', 'Austria', 'Bangladesh', 'Belarus', 'Belgium', 'Bermuda', 'Bhutan', 'Bolivia', 'Botswana', 'Brazil', 'Bulgaria', 'Cambodia', 'Canada', 'Chile', 'Colombia', 'Costa Rica', 'Croatia', 'Curacao', 'Czechia', 'Denmark', 'Dominican Republic', 'Ecuador', 'Egypt', 'Estonia', 'Eswatini', 'Faroe Islands', 'Finland', 'France', 'Germany', 'Ghana', 'Greece', 'Greenland', 'Guam', 'Guatemala', 'Hong Kong', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Ireland', 'Isle of Man', 'Israel', 'Italy', 'Japan', 'Jersey', 'Jordan', 'Kenya', 'Kyrgyzstan', 'Laos', 'Latvia', 'Lesotho', 'Lithuania', 'Luxembourg', 'Macao', 'Madagascar', 'Malaysia', 'Malta', 'Mexico', 'Mongolia', 'Montenegro', 'Mozambique', 'Myanmar', 'Nepal', 'Netherlands', 'New Zealand', 'Nigeria', 'North Macedonia', 'Northern Mariana Islands', 'Norway', 'Pakistan', 'Palestine', 'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Puerto Rico', 'Reunion', 'Romania', 'Russia', 'San Marino', 'Senegal', 'Serbia', 'Singapore', 'Slovakia', 'Slovenia', 'South Africa', 'South Korea', 'South Sudan', 'Spain', 'Sri Lanka', 'Sweden', 'Switzerland', 'Taiwan', 'Thailand', 'Tunisia', 'Turkey', 'US Virgin Islands', 'Uganda', 'Ukraine', 'United Arab Emirates', 'United Kingdom', 'United States', 'Uruguay', 'Venezuela', 'Vietnam']

    if biome_predict == 1:
        model = load_model(CITY_MODEL_PATH)
        classes = ['Aland', 'Albania', 'American Samoa', 'Andorra', 'Argentina', 'Australia', 'Austria', 'Bangladesh', 'Belgium', 'Bermuda', 'Bhutan', 'Bolivia', 'Botswana', 'Brazil', 'Bulgaria', 'Cambodia', 'Canada', 'Chile', 'China', 'Colombia', 'Costa Rica', 'Croatia', 'Curacao', 'Czechia', 'Denmark', 'Dominican Republic', 'Ecuador', 'Egypt', 'Estonia', 'Eswatini', 'Faroe Islands', 'Finland', 'France', 'Germany', 'Ghana', 'Gibraltar', 'Greece', 'Greenland', 'Guam', 'Guatemala', 'Hong Kong', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Iraq', 'Ireland', 'Isle of Man', 'Israel', 'Italy', 'Japan', 'Jersey', 'Jordan', 'Kenya', 'Kyrgyzstan', 'Laos', 'Latvia', 'Lebanon', 'Lesotho', 'Lithuania', 'Luxembourg', 'Macao', 'Madagascar', 'Malaysia', 'Malta', 'Martinique', 'Mexico', 'Monaco', 'Mongolia', 'Montenegro', 'Myanmar', 'Netherlands', 'New Zealand', 'Nigeria', 'North Macedonia', 'Northern Mariana Islands', 'Norway', 'Pakistan', 'Palestine', 'Paraguay', 'Peru', 'Philippines', 'Pitcairn Islands', 'Poland', 'Portugal', 'Puerto Rico', 'Qatar', 'Reunion', 'Romania', 'Russia', 'San Marino', 'Senegal', 'Serbia', 'Singapore', 'Slovakia', 'Slovenia', 'South Africa', 'South Georgia and South Sandwich Islands', 'South Korea', 'Spain', 'Sri Lanka', 'Svalbard and Jan Mayen', 'Sweden', 'Switzerland', 'Taiwan', 'Tanzania', 'Thailand', 'Tunisia', 'Turkey', 'US Virgin Islands', 'Uganda', 'Ukraine', 'United Arab Emirates', 'United Kingdom', 'United States', 'Uruguay', 'Vietnam']

    img_model_sized = cv2.resize(img_resized, (IMAGE_RESIZE[1],IMAGE_RESIZE[0]))
    img_processed = np.expand_dims(img_model_sized,axis=0)

    result = model.predict(img_processed)
    first_index = np.argmax(result)


    first_guess = classes[first_index]
    first_proba = result[0][first_index]

    result_silver = np.delete(result, first_index)
    classes.pop(first_index)

    second_index = np.argmax(result_silver)
    second_guess = classes[second_index]
    second_proba = result_silver[second_index]

    result_bronze = np.delete(result_silver, second_index)
    classes.pop(second_index)

    third_index = np.argmax(result_bronze)
    third_guess = classes[third_index]
    third_proba = result_bronze[third_index]

    return first_guess, second_guess, third_guess, str(first_proba), str(second_proba), str(third_proba)
