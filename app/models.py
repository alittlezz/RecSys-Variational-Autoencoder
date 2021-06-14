# models.py
from datetime import datetime

from flask_login import UserMixin
from sqlalchemy.orm import relationship, backref

from . import db


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    name = db.Column(db.String(1000))
    movies = db.relationship("Movie",
                             secondary="rating")


class Movie(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(1000), unique=True)
    categories = db.Column(db.String(1000))
    image_url = db.Column(db.String(1000))

    users = db.relationship("User",
                             secondary="rating")


class Rating(db.Model):
    userID = db.Column(db.Integer, db.ForeignKey('user.id'), primary_key=True)
    movieID = db.Column(db.Integer, db.ForeignKey('movie.id'), primary_key=True)
    value = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.now())

    user = relationship(User, backref=backref("rating", cascade="all, delete-orphan"))
    movie = relationship(Movie, backref=backref("rating", cascade="all, delete-orphan"))
