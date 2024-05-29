import pickle
import unittest
import numpy as np
import pandas as pd


class TestRecommendFunction(unittest.TestCase):

    def recommend(self, movies: list, genres: list, n: int):
        recommended_movies = []

        for movie, genre in zip(movies, genres):
            index = movies[movies['title'] == movie].index
            if len(index) > 0:
                index = index[0]
                distance = sorted(list(enumerate(self.similarity[index])), reverse=True, key=lambda vector: vector[1])
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

    def setUp(self):
        data = {
        'title': ["The Lion King", "Toy Story", "Frozen", "Moana", "Finding Nemo"],
        'genre': ["Animation,Adventure", "Animation,Comedy", "Animation,Musical", "Animation,Adventure", "Animation,Adventure"]
        }
        self.new_data = pd.DataFrame(data)

        # Mock similarity matrix
        self.similarity = np.array([
            [1, 0.9, 0.8, 0.7, 0.6],
            [0.9, 1, 0.85, 0.75, 0.65],
            [0.8, 0.85, 1, 0.95, 0.55],
            [0.7, 0.75, 0.95, 1, 0.45],
            [0.6, 0.65, 0.55, 0.45, 1]
        ])

        # Ground truth for testing
        ground_truth = {
            "The Lion King": ["Moana", "Finding Nemo"],
            "Toy Story": ["Frozen", "The Lion King"]
        }
        
        movies = pickle.load(open("movies_list.pkl", 'rb'))
        self.similarity = pickle.load(open("similarity.pkl", 'rb'))
        self.movies_list=movies['title'].values
        self.movies = ["The Lion King", "Toy Story", "Frozen", "Moana", "Finding Nemo"]
        self.genres = ["Animation"]
        self.ground_truth = ground_truth

    def test_precision(self):
        # Calculate precision for each movie in ground truth
        precisions = []
        for movie, expected in self.ground_truth.items():
            recommendations = self.recommend([movie], ["Animation"], len(expected))
            true_positives = len(set(recommendations) & set(expected))
            precisions.append(true_positives / len(recommendations) if recommendations else 0)
        
        average_precision = np.mean(precisions)
        self.assertGreaterEqual(average_precision, 0.5, f"Precision too low: {average_precision}")

    def test_recall(self):
        # Calculate recall for each movie in ground truth
        recalls = []
        for movie, expected in self.ground_truth.items():
            recommendations = self.recommend([movie], ["Animation"], len(expected))
            true_positives = len(set(recommendations) & set(expected))
            recalls.append(true_positives / len(expected))
        
        average_recall = np.mean(recalls)
        self.assertGreaterEqual(average_recall, 0.5, f"Recall too low: {average_recall}")

    def test_f1_score(self):
        # Calculate F1 score for each movie in ground truth
        f1_scores = []
        for movie, expected in self.ground_truth.items():
            recommendations = self.recommend([movie], ["Animation"], len(expected))
            true_positives = len(set(recommendations) & set(expected))
            precision = true_positives / len(recommendations) if recommendations else 0
            recall = true_positives / len(expected) if expected else 0
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0
            f1_scores.append(f1)
        
        average_f1_score = np.mean(f1_scores)
        self.assertGreaterEqual(average_f1_score, 0.5, f"F1 score too low: {average_f1_score}")

if __name__ == '__main__':
    unittest.main()
