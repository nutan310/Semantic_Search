import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Page config
st.set_page_config(
    page_title="Semantic Search Engine",
    page_icon="🔍",
    layout="centered"
)

st.title(" 🔍 FAQ Semantic Search Engine")
st.write("Ask a question and get the most relevant answer")

# load model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

#Documents
docs = [
    "King and queen live in a palace",
    "Artificial Intelligence is the future",
    "Virat kohli is a cricket player",
    "Python is used for programming"
]

# document embeddings
doc_embeddings = model.encode(docs)


query = st.text_input("Enter Your Query")

if st.button("🔍 Search") and query:
    # Query Embedding
    query_embedding = model.encode([query])

    #Similarity Scores
    scores = cosine_similarity(
        query_embedding, 
        doc_embeddings)

    #Best Match
    best_index = np.argmax(scores)
    similarity_score = scores[0][best_index]

    st.write("### Best Match")
    st.write(docs[best_index])

    st.write("### Similarity Score")
    print(scores)
    print(best_index)
    st.write(round(scores[best_index]))