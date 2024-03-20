from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import tensorflow as tf
import logging


logger = logging.getLogger("uvicorn")
logger.setLevel(logging.DEBUG)

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

model = tf.keras.models.load_model("trained_model.h5")

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to the API!"}

@app.get("/favicon.ico")
async def favicon():
    return

class InputData(BaseModel):
    externalStatus: str

@app.post("/predict/")
async def predict_status(data: InputData):
    logger.debug(f"Received request with data: {data}")
    text = [data.externalStatus]
    sequence = tokenizer.texts_to_sequences(text)
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=100)
    prediction = model.predict(padded_sequence)
    predicted_label = int(np.argmax(prediction)) 
    logger.debug(f"Predicted label: {predicted_label}")
    return {"predicted_internal_status": predicted_label}
