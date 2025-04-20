'''
TMDB API Data Retrieval Tool

High-Level Goal:
Leveraging the TMDB API tool to retrieve: 1) Budget Revenue; 2) Runtime; 3) Genres; 4) Release Date;
                                          5) Production Companies; 6) Director.

Low-Level Description: 1) Call TMDB API, make sure to include 'credentials.py' for API key; 
                       2) Collect necessary data from 'popular', 'movie_id', and '{movie_id}/credits'. 
                       3) Clean data for 0's and empty entries. This can be done while collecting the data.
                       4) Store cleaned data frame in CSV file.

Notes: First step in Movie Revenue Prediction project. After successfully collecting data, I will train an AI model
       to predict future revenut based on given inputs.
'''

import credentials # Holds API Key
import requests # Need for Application Programming Interface; TMDB API
import pandas as pd # Data Frame Management
from tqdm import tqdm # For progress visualization.

api_key = credentials.api_key # The API key is free for everyone (No Commercial Use) https://developer.themoviedb.org/docs/getting-started
base_api_url = 'https://api.themoviedb.org/3' # Concatenate to "/movie/popular' '/movie/{movie_id} '/movie/{movie_id}/credits'

def collectMovieIds(pages):
    # Call the TMDB API and retrieve a list of popular movies (based on TMDB ratings).
    ids = []
    for page in range(1, pages + 1):
        popular_movies_url = f"{base_api_url}/movie/popular?api_key={api_key}&language=en-US&page={page}"
        response = requests.get(popular_movies_url)

        # Collect and store Movie IDs for the first 100 pages of popular TMDB movies.
        movies = response.json()['results']
        ids.extend(movie['id'] for movie in movies)
    return ids

def get_movie_details(movie_id):
    url = f"{base_api_url}/movie/{movie_id}?api_key={api_key}&language=en-US"
    return requests.get(url).json()

def get_movie_credits(movie_id):
    url = f"{base_api_url}/movie/{movie_id}/credits?api_key={api_key}&language=en-US"
    return requests.get(url).json()

# A very robust, yet slow method to checking for invalid data entries.
def is_valid_movie(movie_data):
    if not movie_data.get('title'):
        return False
    if not movie_data.get('budget') or movie_data['budget'] == 0:
        return False
    if not movie_data.get('revenue') or movie_data['revenue'] == 0:
        return False
    if not movie_data.get('runtime') or movie_data['runtime'] == 0:
        return False
    if not movie_data.get('release_date'):
        return False
    return True

# Main Processing
pages = 100
movie_ids = collectMovieIds(pages)
data = []

for movie_id in tqdm(movie_ids):
    details = get_movie_details(movie_id)
    credits = get_movie_credits(movie_id)

    director = next((member['name'] for member in credits['crew'] if member['job'] == 'Director'), None)

    movie_data = {
        'title': details.get('title'),
        'budget': details.get('budget'),
        'revenue': details.get('revenue'),
        'runtime': details.get('runtime'),
        'genres': ", ".join([genre['name'] for genre in details.get('genres', [])]),
        'release_date': details.get('release_date'),
        'production_companies': ", ".join([company['name'] for company in details.get('production_companies', [])]),
        'director': director
    }
    if is_valid_movie(movie_data):
        data.append(movie_data)

# Convert to DataFrame
df = pd.DataFrame(data)
df.to_csv('movie_dataset.csv', index=False)

print(df.head())