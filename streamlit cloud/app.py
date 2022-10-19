from grab_embeddings import EmbeddingCalculator
import faiss
import os
import requests
import tarfile 
from PIL import Image
import base64
import requests
import streamlit as st
import io


DIRECTORY = os.path.abspath(os.path.dirname(__file__))
ARTIFACT_DIR = os.path.join(DIRECTORY, 'S3artifacts')

@st.cache
def fetch_data(url, data_dir, download=False):
    if download:
        response = requests.get(url, stream=True)
        file = tarfile.open(fileobj=response.raw, mode="r|gz")
        file.extractall(path=data_dir)

@st.cache(allow_output_mutation=True)
def load_artifacts(url):
    # make directory for artifacts if one isn't made
    try:
        os.mkdir(ARTIFACT_DIR)
        # grab model and index artifacts
        fetch_data(url, ARTIFACT_DIR, download = True)
    except FileExistsError:
        pass

    # load the artifacts in
    analyzer = EmbeddingCalculator(os.path.join(ARTIFACT_DIR, "model_pets.pt"),  37)     # 37 classes for our dataset
    index = faiss.read_index(os.path.join(ARTIFACT_DIR, "pet_faiss_index"))

    return analyzer, index

pets_url = 'https://exp-instance-retrieval.s3.amazonaws.com/stem-goes-red/artifacts/artifacts.tar.gz'
ANALYZER, INDEX = load_artifacts(pets_url)

# search for the top_k most similar images
def preprocess_input(target_img_bytes, top_k = 12):
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


def set_state_if_absent(key, value):
    if key not in st.session_state:
        st.session_state[key] = value

def main():

    set_state_if_absent("results", None)
    set_state_if_absent("image_input", None)

    st.header(":dog: Find Your Pet :cat:")
    st.write("""Upload a picture of your furry friend and we'll search for cats or dogs just like them.""")
    # with st.sidebar:
    #     st.sidebar.write("## Upload Image")
    with st.form(key = "image_form" ):
        image_input = st.file_uploader("", type=["png", "jpg", "jpeg"], accept_multiple_files=False)
        submit_upload = st.form_submit_button("Submit & Search")
        st.session_state.submitted_upload = submit_upload
    
    
    n_cols = 4

    if submit_upload: 
        if image_input is not None:   
            st.session_state.image_input = image_input
            st.write("### You Uploaded:")
            st.image(Image.open(st.session_state.image_input))

            with st.spinner("⌛️ &nbsp;&nbsp; Searching..."):
                encoded_img = base64.b64encode(image_input.getvalue()).decode('utf-8')
                st.session_state.results = preprocess_input(encoded_img, 12)
                # images = [Image.open(item) for item in st.session_state.results['best_paths']]
                # get from S3
                images = [Image.open(requests.get(item, stream = True).raw) for item in st.session_state.results['best_paths']]
                images_resized = [im.resize((224,224)) for im in images]
                st.header("Pets like yours!")
                n_rows = 1 + len(images_resized) // int(n_cols)
                rows = [st.container() for _ in range(n_rows)]
                cols_per_row = [r.columns(n_cols) for r in rows]
                cols = [column for row in cols_per_row for column in row]

            
                for image_index, cat_image in enumerate(images_resized):
                    cols[image_index].image(cat_image)
        # st.image(images_resized, use_column_width=True)
        else:
            st.error("You didn't upload a picture")
    
    # run_search = st.button("Find Similar Images!")
    # st.session_state.run_search = run_search






main()