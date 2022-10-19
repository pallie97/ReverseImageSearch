from typing import List, Dict, Any, Tuple

import os
import logging
import requests
from time import sleep
from uuid import uuid4
import streamlit as st
import json

import torch
import matplotlib.pyplot as plt
import numpy as np


API_ENDPOINT = os.getenv("API_ENDPOINT", "http://localhost:8000")
IMAGE_UPLOAD = "upload-file"
SEARCH_REQUEST = "search"


def upload_image(file):
    url = f"{API_ENDPOINT}/{IMAGE_UPLOAD}"
    file = {"file": file}
    response = requests.post(url, files=file).json()
    return response


def search(image, top_k):
    url = f"{API_ENDPOINT}/{SEARCH_REQUEST}"
    req = json.dumps({"image_bytes": image, "top_k": top_k})
    response = requests.post(url, data=req).json()
    return response