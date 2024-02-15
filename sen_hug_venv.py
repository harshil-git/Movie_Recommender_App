import streamlit as st
import pickle
import requests
import os
import numpy as np
import pandas as pd
import faiss


from sentence_transformers import SentenceTransformer


model = SentenceTransformer('all-mpnet-base-v2')



def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=a970be7f0ac3aaa7334214a23ea19201".format(movie_id)
    data_req=requests.get(url)
    data_req=data_req.json()
    poster_path = data_req['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/"+poster_path
    return full_path

movies = pickle.load(open("movie.pkl", 'rb'))
similarity = pickle.load(open("faiss_index.pkl", 'rb'))

movies_list=movies['title'].values

st.header("Movie Recommender System")
selectvalue=st.selectbox("Select movie from dropdown", movies_list)


def recommend(movie):
    svec = np.array(model.encode(movie)).reshape(1,-1)
    distance, ind = similarity.search(svec, k=5)
    recommend_5_title = list(movies.loc[ind[0],'title'])
    recommend_5_ids = list(movies.loc[ind[0],'id'])
    recommend_poster = []
    for i in recommend_5_ids:
        recommend_poster.append(fetch_poster(i))
    return recommend_5_title, recommend_poster



if st.button("Show Recommend"):
    movie_name, movie_poster = recommend(selectvalue)
    col1,col2,col3,col4,col5=st.columns(5)
    with col1:
        st.text(movie_name[0])   
        st.image(movie_poster[0]) 
    with col2:
        st.text(movie_name[1]) 
        st.image(movie_poster[1])  
    with col3:
        st.text(movie_name[2]) 
        st.image(movie_poster[2])   
    with col4:
        st.text(movie_name[3])  
        st.image(movie_poster[3])   
    with col5:
        st.text(movie_name[4])
        st.image(movie_poster[4])
        
    
