from pathlib import Path

from fastapi import APIRouter, UploadFile, File

from pydantic import BaseConfig

from api.controller.indexer import ANALYZER, INDEX
from api.schema import SearchRequest, SearchResponse
import os
import io
import base64
BaseConfig.arbitrary_types_allowed = True

router = APIRouter()
import numpy as np
from PIL import Image
import uuid

DIRECTORY = os.getcwd()

# store locally if you want
# not currently in use
@router.post("/upload-file")
def get_image(file: UploadFile = File(...)):
    image = Image.open(file.file)
    name = str(uuid.uuid4())+".jpg"
    image.save(fr"{DIRECTORY}\\ui\\storage\{name}")
    return {"name": name}


# user uploads image and assigns top_k results
@router.post("/search")
def search(request: SearchRequest):
    response = _preprocess_input(request.image_bytes, request.top_k)
    return response


def _preprocess_input(target_img_bytes, top_k) -> SearchResponse:
    img_bytes = base64.decodebytes(target_img_bytes.encode('utf-8'))
    target_img = Image.open(io.BytesIO(img_bytes))
    target_tranformed = ANALYZER.transform(target_img)
    target_embeddings =ANALYZER.getEmbeddings(target_tranformed)

    distance, indices = INDEX.search(target_embeddings.numpy(), top_k)

    data_path = DIRECTORY + "\\new_id\\"
    # best_paths = [data_path+str(i)+".jpg" for i in indices[0]]    # grab locally
    # grab from S3
    best_paths = [f"https://exp-instance-retrieval.s3.amazonaws.com/stem-goes-red/new_id/{i}.jpg" for i in indices[0]]
    
    return {"best_paths": best_paths}