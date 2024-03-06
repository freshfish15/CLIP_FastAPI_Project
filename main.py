import os
from fastapi import FastAPI, File, UploadFile
import torch
from PIL import Image
import base64
import open_clip
from io import BytesIO
import shutil
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

app = FastAPI()

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')


class ImageText(BaseModel):
    text: str
    imageBase64: str


# Define the path where you want to store the uploaded files
UPLOAD_DIRECTORY = "uploaded_image"
# Ensure the upload directory exists
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.post("/uploadImage")
async def upload_image(text: str, file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)

        # Open the file in write mode (binary) and save the uploaded file to the filesystem
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Split the input string by comma into a list of strings
        string_list = text.split(',')

        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        tokenizer = open_clip.get_tokenizer('ViT-B-32')

        image_data = file.file.read()

        image = preprocess(Image.open(file_path)).unsqueeze(0)
        # text = tokenizer(["a diagram", "a cat", "a truck"])
        text = tokenizer(string_list)

        # with torch.no_grad(), torch.cuda.amp.autocast(), torch.cpu.amp.autocast():

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        probs_list = text_probs.tolist()
        decimal_probs_list = [f"{value:.20f}" for value in probs_list[0]]

        combined_dict = dict(zip(string_list, decimal_probs_list))

        os.remove(file_path)
        # return("label: ", string_list, "Label probs:", decimal_probs_list)  # prints: [[1., 0., 0.]]
        return JSONResponse(content=combined_dict)

    except Exception as e:
        return str(e)


@app.post("/uploadImageBase64")
async def upload_image_base64(imagetext: ImageText):
    try:
        filename = "temp.png"
        # file_path = os.path.join(UPLOAD_DIRECTORY, filename)

        # Add proper padding, if necessary
        # if len(temp)%3 == 1:
        #     temp += "=="
        # elif len(temp)%3 == 2:
        #     temp += "="

        # image_data_frombase64 = base64.decodebytes(imagetext.imageBase64)

        # Open the file in write mode (binary) and save the uploaded file to the filesystem

        #
        # with open(file_path, "wb") as buffer:
        #     shutil.copyfileobj(image_data_frombase64, buffer)
        byte_data = base64.b64decode(imagetext.imageBase64)
        image_data = BytesIO(byte_data)
        img = Image.open(image_data)
        image_path = './output_image.png'
        if image_path:
            img.save(image_path)
        # with open('output_image.png', 'wb') as file:
        #     file.write(base64.decodex(imagetext.imageBase64))

        file_path = os.path.join(UPLOAD_DIRECTORY, 'output_image.png')

        # Split the input string by comma into a list of strings
        string_list = imagetext.text.split(',')

        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        tokenizer = open_clip.get_tokenizer('ViT-B-32')

        # image_data = image_data_frombase64.read()

        image = preprocess(Image.open(image_path)).unsqueeze(0)
        # text = tokenizer(["a diagram", "a cat", "a truck"])
        text = tokenizer(string_list)

        # with torch.no_grad(), torch.cuda.amp.autocast(), torch.cpu.amp.autocast():

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        probs_list = text_probs.tolist()
        decimal_probs_list = [f"{value:.20f}" for value in probs_list[0]]

        combined_dict = dict(zip(string_list, decimal_probs_list))

        # os.remove(file_path)
        # return("label: ", string_list, "Label probs:", decimal_probs_list)  # prints: [[1., 0., 0.]]
        return JSONResponse(content=combined_dict)

    except Exception as e:
        return str(e)


def decode_base64(data):
    # Check if there is padding
    missing_padding = len(data) % 4
    if missing_padding:
        data += '=' * (4 - missing_padding)  # Add the required padding
    return base64.b64decode(data)
