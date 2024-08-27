import pandas as pd
from fastapi import FastAPI, UploadFile, File
from modules.model.predict import predict_city_countryside
import os
import shutil

app = FastAPI()

## Load the model or the 2 models we'll need
#app.state.model = load_model()

###################################################""

    ## INPUT
## Upload image.jpeg from a screenshot

@app.post("/predict_country")
def predict_country_api():
    img = "../../data/574db65a1289dcd5e7f92b0674fac33a.jpg"
    result = predict_city_countryside(img)
    return {f"message" : "Predicting the country : {result}"}


## Send image.jpeg to preprocess.py

        ## image.jpeg transform into array RGB

        ## predict city/countryside

## image.jpeg goes to either city or countryside model to predict

    ##OUTPUT
## Predict the country of image.jpeg

@app.get("/")
def index():
    return {"message": "API is running!"}
