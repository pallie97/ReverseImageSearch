# from distutils.command.upload import upload
import streamlit as st
from utils import upload_image, search
from PIL import Image
import base64
import requests

def set_state_if_absent(key, value):
    if key not in st.session_state:
        st.session_state[key] = value



def main():

    
    set_state_if_absent("results", None)
    set_state_if_absent("image_input", None)
    # set_state_if_absent("run_search", None)

    st.header(":dog: Find Your Pet :cat:")
    st.write("""Upload a picture of your furry friend and we'll search for cats or dogs just like them.""")
    with st.sidebar:
        st.sidebar.write("## Upload Image")
        with st.form(key = "image_form" ):
            image_input = st.file_uploader("", type=["png", "jpg"], accept_multiple_files=False)
            submit_upload = st.form_submit_button("Submit")
            st.session_state.submitted_upload = submit_upload
            

    if submit_upload: 
        if image_input is not None:   
            st.sidebar.success("Upload Successful ✅")
            # raw_json = upload_image(image_input)
            st.session_state.image_input = image_input
            st.write("### You Uploaded:")
            st.image(Image.open(st.session_state.image_input))
        else:
            st.sidebar.error("You didn't upload a picture")
    
    run_search = st.button("Find Similar Images!")
    st.session_state.run_search = run_search

    n_cols = 4
    if run_search and image_input:
        st.image(Image.open(st.session_state.image_input))
        encoded_img = base64.b64encode(image_input.getvalue()).decode('utf-8')
        st.session_state.results = search(encoded_img, 12)
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



main()