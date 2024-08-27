import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from modules.model.predict import predict_city_countryside
import os
import shutil

                    ### TO DO ###

## Consider HTTP request to control we are only getting .jpeg
## The return of the API must be JSON

## Alexandre : use Streamlit to handle images (bytes conversion)
    ## or use another way to handle images format

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
## Load the model or the 2 models we'll need
#app.state.model = load_model()

###################################################""

    ## INPUT
## Upload image.jpeg from a screenshot

UPLOAD_DIR = "../../data/uploaded_images/"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/predict_country")
def predict_country_api(file : UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    predicted_country = predict_city_countryside(file_path)

    os.remove(file_path)
    return {"predicted_country": predicted_country}

## Send image.jpeg to preprocess.py
        ## image.jpeg transform into array RGB

""" @app.post("/predict_country")
def predict_country_api(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        predicted_country = predict_city_countryside(file_path)
        os.remove(file_path)
        print(predicted_country)
        return {"predicted_country": predicted_country}

    except Exception as e:
        return {"error": str(e)} """

        ## predict city/countryside

## image.jpeg goes to either city or countryside model to predict

    ##OUTPUT
## Predict the country of image.jpeg



@app.get("/")
def index():
    return {"message": "API is running!"}
