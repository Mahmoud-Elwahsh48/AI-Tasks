import streamlit as st
import pandas as pd
from transformers import BeitImageProcessor, BeitForImageClassification
from transformers import T5ForConditionalGeneration, T5Tokenizer
from happytransformer import HappyTextToText, TTSettings
from nltk.tokenize import word_tokenize

import torch
from transformers import pipeline
from PIL import Image
from nltk.corpus import stopwords

import nltk
nltk.download('punkt')
nltk.download('stopwords')

import gensim
from gensim import corpora
import re
import numpy as np
from scipy.io.wavfile import write

import requests
import uuid
import wave

import os



# Set the page config for the app (only once)
st.set_page_config(page_title="AI models Task ", layout="centered")

# Define the pages as functions
def page_1():

    """Page for identifying objects in text."""


    API_URL = "https://api-inference.huggingface.co/models/dbmdz/bert-large-cased-finetuned-conll03-english"
    headers = {"Authorization": "Bearer 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"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    user_input = st.text_area("Enter any text to identify on object")
    output = query({
        "inputs": user_input,
    })
    if st.button("identify"):
        if user_input:

            data = output
            df = pd.DataFrame(data)
            st.table(df)





def page_2():

    """Page for completing sentences."""

    st.title("Sentence Completion")

    API_URL = "https://api-inference.huggingface.co/models/google-bert/bert-base-uncased"
    headers = {"Authorization": "Bearer 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"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    user_input = st.text_area("Enter any text to complete remains sentence")
    output = query({
        "inputs": user_input,
    })
    if st.button("complete"):
        if user_input:

            data = output
            df = pd.DataFrame(data)
            st.table(df)




def page_3():

    """Page for converting speech to text."""







    # Hugging Face API details
    API_URL = "https://api-inference.huggingface.co/models/facebook/wav2vec2-large-960h"
    headers = {"Authorization": "Bearer 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"}




    def record_audio():
        st.title("Audio Recorder")

        audio_data = st.experimental_audio_input("Record Audio")

        if audio_data:
            st.audio(audio_data)

            if audio_data is not None:
                st.success("Recording completed!")


                # Save the audio if needed
                file_path = "recorded_audio.wav"

                try:
                    if hasattr(audio_data, "getvalue"):
                        with open(file_path, "wb") as wf:
                            wf.write(audio_data.getvalue())  # Save the recorded audio to a file
                    else:
                        st.error("No audio data found.")

                    # Check if the file has been created successfully
                    if os.path.exists(file_path):
                        st.write(file_path)  # Play back the audio file

                        # Send audio to API
                        with open(file_path, "rb") as file:
                            audio_content = file.read()  # Read the file contents
                            response = requests.post(API_URL, headers=headers, data=audio_content)

                            # Handle API response
                            if response.status_code == 200:
                                output = response.json()  # Assuming the API returns JSON
                                st.write(f"Extracted from speech record: {output.get('text', 'No text found').lower()}")
                            else:
                                st.error(f"API request failed with status code {response.status_code}: {response.text}")
                    else:
                        st.error("Failed to save audio. No audio recorded.")

                except Exception as e:
                    st.error(f"An error occurred: {e}")

                    # Provide download option
                if os.path.exists(file_path):
                    with open(file_path, "rb") as file:
                        st.download_button(
                            label="Download Recorded Audio",
                            data=file,
                            file_name="recorded_audio.wav",
                            mime="audio/wav"
                        )

    record_audio()
    def query():
            uploaded_file = st.file_uploader("Choose an speech file", type=["wav", "mp3"])
            if uploaded_file is not None:  # Check if a file has been uploaded
                # Display the file name
                st.write("Uploaded file:", uploaded_file.name)

                # If it's a WAV file, you can read it as follows
                if uploaded_file.type == "audio/wav":
                    # Read the WAV file
                    data = uploaded_file.read()  # Read file content as bytes
                    response = requests.post(API_URL, headers=headers, data=data)
                    output=response.json()
                    st.write(f"extracted from audio file : {output}")




                # If it's an MP3 file, you can use an appropriate library to handle it
                elif uploaded_file.type == "audio/mpeg":
                    data = uploaded_file.read()  # Read file content as bytes
                    response = requests.post(API_URL, headers=headers, data=data)


                    output=response.json()
                    st.write(f"extracted from audio file : {output}")

    #recordaudio()
    query()


def page_4():
    """Page for object detection and image captioning."""


    def image_caption():

           st.title("Upload and Display Image for image caption")

           API_URL = "https://api-inference.huggingface.co/models/nlpconnect/vit-gpt2-image-captioning"
           headers = {"Authorization": "Bearer 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"}

           uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])
           if uploaded_file is not None:

               file_bytes = uploaded_file.read()  # Read file as bytes
               response = requests.post(API_URL, headers=headers, data=file_bytes)
               output= response.json()
               st.write(f"image caption text: {output[0]['generated_text']}")

           else:
                st.warning("Please upload a file.")
    image_caption()


    def objectdetection():
       st.title("Upload and Display Image for Object Detection")
       uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"],key="image_uploader_1")
       if uploaded_file is not None:
           image = Image.open(uploaded_file)
           st.image(image, caption="Uploaded Image", use_column_width=True)
           processor = BeitImageProcessor.from_pretrained('microsoft/beit-large-patch16-224-pt22k-ft22k')
           model = BeitForImageClassification.from_pretrained('microsoft/beit-large-patch16-224-pt22k-ft22k')

           inputs = processor(images=image, return_tensors="pt")

           outputs = model(**inputs)
           logits = outputs.logits

           # model predicts one of the 21,841 ImageNet-22k classes
           predicted_class_idx = logits.argmax(-1).item()
           st.write("Predicted class:", model.config.id2label[predicted_class_idx])
       else:
           st.write("Please upload an image.")
    objectdetection()


def page_5():

    """Page for extracting topics from text."""

    st.title("Text Extraction")
    # Text area for user input
    user_input = st.text_area("Enter your documents (each document on a new line):", height=200)
    # Preprocessing function
    def preprocess_text(text):
        # Lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        return tokens

    # Process the input when the button is clicked
    if st.button("Process Documents"):
        # Split the input into a list of documents
        documents = user_input.split('\n')

        # Preprocess each document
        processed_docs = [preprocess_text(doc) for doc in documents]
        # Create a dictionary representation of the documents
        dictionary = corpora.Dictionary(processed_docs)

        # Create a document-term matrix
        doc_term_matrix = [dictionary.doc2bow(doc) for doc in processed_docs]

        # Create the LDA model
        lda_model = gensim.models.ldamodel.LdaModel(doc_term_matrix, num_topics=3, id2word=dictionary, passes=15)

        # Print the topics with the top words
        topics = lda_model.print_topics(num_words=4)
        for topic in topics:
            st.write(f"Topic {topic[0]}: {topic[1]}")




def page_6():

    """Page for answering questions based on context."""

    st.title("Text to reply on Question Answering")

    API_URL = "https://api-inference.huggingface.co/models/distilbert/distilbert-base-cased-distilled-squad"
    headers = {"Authorization": "Bearer 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"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()
    user_input1 = st.text_area("Enter original text")

    user_input = st.text_area("Enter questions to can answer")


    output = query({
        "inputs": {
            "question": user_input,
            "context": user_input1
        },
    })
    if st.button("answer"):
        if user_input:
            data = output
            df = pd.DataFrame(data,index=[0])
            st.table(df)


def page_7():
    """Page for language translation and grammar correction."""

    def Grammar_Correction():

        st.title("Grammar_Correction")


        happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")

        args = TTSettings(num_beams=5, min_length=1)


        # User input for text
        user_input = st.text_area("Enter text to correct grammar:")

        if st.button("Correct Grammar"):
            if user_input:
                # Add the prefix "grammar: " before each input
                result = happy_tt.generate_text(user_input, args=args)

                st.write(result.text)  # This sentence has bad grammar.
            else:
                st.write("Please enter some text to correct.")

    Grammar_Correction()
    def language_translation():

        API_URL = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-ar"
        headers = {"Authorization": "Bearer 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"}

        def query(payload):
            response = requests.post(API_URL, headers=headers, json=payload)
            return response.json()

        # User input for text
        user_input = st.text_area("Enter text to translate sentence:")

        if st.button("translate"):
            if user_input:
                output = query({
                    "inputs": user_input,
                })
                st.write(output[0]['translation_text'])


            else:
                st.write("Please enter some text to translate.")
    language_translation()

def homepage():

    st.image("https://via.placeholder.com/800x150.png?text=AI+Models+Task", use_column_width=True)
    st.title("Welcome to the AI Task Hub")
    st.write("""
    Explore various AI technologies below. Click on each task to learn more or interact with demos.
    """)





    # Create small points for each AI task
    st.markdown("""
       ### AI Tasks Overview
       - **Identify Objects**: Identify objects in a text input.
       - **Sentence Completion**: Complete incomplete sentences.
       - **Speech-to-Text Conversion**: Convert speech or audio files to text.
       - **Object Detection & Image Captioning**: Detect objects and generate captions for images.
       - **Text Extraction**: Extract topics from documents.
       - **Question Answering**: Answer questions based on provided context.
       - **Language Translation & Grammar Correction**: Translate text and correct grammar.
       """)


    st.write("Explore each of these tasks in more detail by selecting from the dropdown menu.")


# Dropdown menu in the upper right corner
st.markdown(
    """
    <style>
    .dropdown-container {
        position: absolute;
        top: 10px;
        right: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Wrap the dropdown in a div to apply the CSS class
st.markdown('<div class="dropdown-container">', unsafe_allow_html=True)
st.sidebar.title("AI Tasks")

page = st.sidebar.selectbox("Select any AI Models", ["Homepage", "identify to object", "Sentence Completion", "conversion speech to text",
                                      "Object Detection and image caption", "Text Extraction", "Questions Answering", "Language Translation and Grammar Correction Tool "])
st.markdown('</div>', unsafe_allow_html=True)

# Run the appropriate page based on selection
if __name__ == "__main__":
    if page == "Homepage":
        homepage()
    elif page == "identify to object":
        page_1()
    elif page == "Sentence Completion":
        page_2()
    elif page == "conversion speech to text":
        page_3()
    elif page == "Object Detection and image caption":
        page_4()

    elif page == "Text Extraction":
        page_5()
    elif page == "Questions Answering":
        page_6()
    elif page == "Language Translation and Grammar Correction Tool ":
        page_7()
