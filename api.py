# main.py

import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# Import the sample movie data
movies = pickle.load(open("movies_list.pkl", 'rb'))
similarity = pickle.load(open("similarity.pkl", 'rb'))
movies_list=movies['title'].values

app = FastAPI()

class SuggestRequest(BaseModel):
    genres: List[str]
    movies: List[int]

class SuggestResponse(BaseModel):
    movie_ids: List[int]

@app.get("/suggest", response_model=SuggestResponse)
async def suggest_movies(request: SuggestRequest):
    return SuggestResponse(movie_ids=recommend(movies, request.genres, request.movies, 20));

def recommend(movies: list, genres: list, n: int):
    recommended_movies = []

    for movie, genre in zip(movies, genres):
        index = movies[movies['title'] == movie].index
        if len(index) > 0:
            index = index[0]
            distance = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda vector: vector[1])
            i = 0
            matched = 0
            while matched < n and i < len(distance):
                movie_data = movies.iloc[distance[i][0]]
                genre_list = movie_data.genre.split(',')
                if genre in genre_list:
                    recommended_movies.append(movie_data.id)
                    matched += 1
                i += 1

    return recommended_movies