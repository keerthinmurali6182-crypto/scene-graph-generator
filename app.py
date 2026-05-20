import streamlit as st
from PIL import Image
# Import your internal logic here
# from src.engine_models import your_inference_function 

st.title("Scene Graph Generator")

# A file uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Generate Scene Graph'):
        with st.spinner('Processing...'):
            # Call your existing logic here:
            # results = your_inference_function(image)
            
            # Display results
            st.write("Scene Graph Results:")
            # st.json(results) 
            st.success("Inference complete!")