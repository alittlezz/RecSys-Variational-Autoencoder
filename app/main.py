# main.py

import torch
from flask import Blueprint, render_template, request, redirect, url_for
from flask_login import login_required, current_user

from . import db
from .models import Movie, Rating
from .vae import DualVAE

main = Blueprint('main', __name__)

# Init all

import pandas as pd
from datetime import datetime
import numpy as np
import torch


def RMSE(ratings_pred, ratings):
    ratings_mask = ratings > 0
    return torch.sum((ratings_pred * ratings_mask - ratings) ** 2)


def generate(batch_size, device, data_in, data_out=None, shuffle=False, samples_perc_per_epoch=1):
    yield Batch(device, [], data_in, data_out)


class Batch:
    def __init__(self, device, idx, data_in, data_out=None):
        self._device = device
        self._idx = idx
        self._data_in = data_in
        self._data_out = data_out

    def get_ratings(self, is_out=False):
        data = self._data_in
        return data

    def get_ratings_to_dev(self, is_out=False):
        ratings = self.get_ratings(is_out)
        return torch.Tensor(ratings.toarray()).to(self._device)


# Load data into memory

df_train = pd.read_csv('data/movielens1m_test.csv')
df_test = pd.read_csv('data/movielens1m_train.csv')
df_train['UserID'] = df_train['UserID']
df_test['UserID'] = df_test['UserID']
USERS = max(df_train['UserID'].max(), df_test['UserID'].max()) + 1
df_train['MovieID'] = df_train['MovieID']
df_test['MovieID'] = df_test['MovieID']
MOVIES = max(df_train['MovieID'].max(), df_test['MovieID'].max()) + 1

df = pd.concat([df_train, df_test])
ROWS = df['UserID']
COLS = df['MovieID']
print('Finished loading data...')

from scipy.sparse import csr_matrix
import random
import numpy as np

seed = 1337
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')
torch.set_default_tensor_type(torch.FloatTensor)

# Construct the train and test rating matrices.

train_data = csr_matrix((df_train['Rating'], (df_train['UserID'], df_train['MovieID'])), shape=(USERS, MOVIES),
                        dtype=np.float32)
RMSE_train = torch.Tensor(train_data.copy().toarray()).to(device)
rows, cols = train_data.nonzero()
train_mn = train_data.min()
train_mx = train_data.max()
for i, j in zip(rows, cols):
    train_data[i, j] = (train_data[i, j] - train_mn) / (train_mx - train_mn)

test_data = csr_matrix((df_test['Rating'], (df_test['UserID'], df_test['MovieID'])), shape=(USERS, MOVIES),
                       dtype=np.float32)
RMSE_test = torch.Tensor(test_data.copy().toarray()).to(device)
rows, cols = test_data.nonzero()
test_mn = test_data.min()
test_mx = test_data.max()
for i, j in zip(rows, cols):
    test_data[i, j] = (test_data[i, j] - test_mn) / (test_mx - test_mn)

print('Finished preprocessing data...')

# Precalculate batches for faster computing.
R_train = train_data
R_train_tensor = []
for i in range(USERS):
    batch = R_train[i].nonzero()[1]
    ts = torch.from_numpy(R_train[i, batch].todense().transpose().astype(np.float32)).to(device)
    R_train_tensor.append(ts)

R_train_T = train_data.transpose()
R_train_tensor_T = []
for j in range(MOVIES):
    batch = R_train_T[j].nonzero()[1]
    ts = torch.from_numpy(R_train_T[j, batch].todense().transpose().astype(np.float32)).to(device)
    R_train_tensor_T.append(ts)
print('Finished preprocessing vectors...')

args = {
    'dataset': '',
    'hidden_dim': 40,
    'latent_dim': 5,
    'batch_size': train_data.count_nonzero(),
    'beta': None,
    'gamma': 1,
    'lr': 1e-3,
    'n_epochs': 150,
    'dropout_rate': 0.8,
    'print_step': 1,
    'n_enc_epochs': 3,
    'n_dec_epochs': 1,
    'not_alternating': True,
}

model_kwargs = {
    'hidden_dim': args['hidden_dim'],
    'latent_dim': args['latent_dim'],
    'input_dim_users': train_data.shape[0],
    'input_dim_movies': train_data.shape[1],
    'device': device
}

model = DualVAE(**model_kwargs).to(device)
model.load_state_dict(torch.load('models/BDCVAE.pt', map_location=device))
model.eval()


@main.route('/')
def index():
    return render_template('index.html')


@main.route('/profile')
@login_required
def profile():
    movie_name = request.args.get('movieName')
    ratings = Rating.query.filter_by(userID=current_user.id).order_by(Rating.created_at.desc()).all()
    rated_movies_id = list(map(lambda rating: rating.movie.id, ratings))
    movies = Movie.query.filter(Movie.id.not_in(rated_movies_id)).all()
    if movie_name:
        ratings = list(filter(lambda rating: movie_name.lower() in rating.movie.name.lower(), ratings))
        movies = list(filter(lambda movie: movie_name.lower() in movie.name.lower(), movies))
    return render_template('profile.html', name=current_user.name, movies=movies[:20], userRatings=ratings,
                           movie_name='' if movie_name is None else movie_name)


@main.route('/rec')
@login_required
def rec():
    return render_template('rec.html')


@main.route('/recme')
@login_required
def rec_me():
    for batch in generate(batch_size=1,
                          device=device,
                          data_in=train_data,
                          data_out=train_data,
                          samples_perc_per_epoch=1
                          ):
        ratings = batch.get_ratings_to_dev()
        user_ratings = Rating.query.filter_by(userID=current_user.id).order_by(Rating.created_at.desc()).all()
        rated_movies_id = list(map(lambda rating: (rating.movie.id, rating.value), user_ratings))
        user_features = torch.zeros(size=(1, train_data.shape[1])).to(device)
        for id, val in rated_movies_id:
            if id < train_data.shape[1]:
                user_features[0, id] = val / 5

        ratings_pred = model(ratings, R_train, R_train_tensor, R_train_T, R_train_tensor_T, userFeatures=user_features, calculate_loss=False)
        for id, val in rated_movies_id:
            if id < train_data.shape[1]:
                ratings_pred[0, id] = -10000

        pred = ratings_pred.detach().numpy().flatten().tolist()
        movies = sorted(range(len(pred)), key=lambda i: pred[i], reverse=True)[:20]
    movies = Movie.query.filter(Movie.id.in_(movies)).all()
    return render_template('rec.html', movies=movies)


@main.route('/ratings', methods=['POST'])
@login_required
def rateMovie():
    movie_id = request.form.get('movieID')
    rating_value = request.form.get('rating')

    rating = Rating(userID=current_user.id, movieID=movie_id, value=rating_value)
    db.session.add(rating)
    db.session.commit()
    return redirect(url_for('main.profile'))
