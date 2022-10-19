from grab_embeddings import EmbeddingCalculator
import faiss
import os
import requests
import tarfile 

# DIRECTORY = os.getcwd()
DIRECTORY = os.path.abspath(os.path.dirname(__file__))
ARTIFACT_DIR = os.path.join(DIRECTORY, 'S3artifacts')


def fetch_data(url, data_dir, download=False):
    if download:
        response = requests.get(url, stream=True)
        file = tarfile.open(fileobj=response.raw, mode="r|gz")
        file.extractall(path=data_dir)

# make directory for artifacts if one isn't made
try:
    os.mkdir(ARTIFACT_DIR)
    # grab model and index artifacts
    pets_url = 'https://exp-instance-retrieval.s3.amazonaws.com/stem-goes-red/artifacts/artifacts.tar.gz'
    fetch_data(pets_url, ARTIFACT_DIR, download = True)
except FileExistsError:
    pass


# load the artifacts in
ANALYZER = EmbeddingCalculator(DIRECTORY+"\\S3artifacts\\model_pets.pt",  37)     # 37 classes for our dataset
INDEX = faiss.read_index(DIRECTORY+ "\\S3artifacts\\pet_faiss_index")