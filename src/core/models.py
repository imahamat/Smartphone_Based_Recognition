from django.db import models

# Create your models here.
import pandas as pd
import numpy as np
import warnings
from django.core.cache import cache
from os import path
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import lightgbm as ltb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.externals import joblib

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class Classifier:
 # Ces chemins seront utilisés pour enregistrer notre modèle ML dans un fichier afin que nous n'ayons pas à le créer à chaque fois que nous voulons faire une action avec lui
 model_filename = '/code/Human_Activities__Recognition/model-cache/cache.pkl'
 model_cache_key = 'model_cache'
 model_rel_path = "Human_Activities__Recognition/model-cache/cache.pkl"

# Nous disposerons également d'un cache permettant un accès plus rapide au modèle.
 score_cache_key = 'score_cache'

# Contructors
 def __init__(self):

     self.lreg = LogisticRegression(solver='newton-cg')
     #self.dtree = DecisionTreeClassifier(criterion="entropy")
     #self.xgb = XGBClassifier()
     #self.ltb = ltb.LGBMClassifier()
     #self.rf = RandomForestClassifier(n_estimators=600, max_depth=300, max_features='sqrt')

# Methods

 # Training the model
 def train_model(self):

     X_train = pd.read_csv('/content/gdrive/My Drive/HAPT Data Set/Train/X_train.txt',sep=' ', header=None)
     y_train = pd.read_csv('/content/gdrive/My Drive/HAPT Data Set/Train/y_train.txt',header=None)
     X_test = pd.read_csv('/content/gdrive/My Drive/HAPT Data Set/Test/X_test.txt',sep=' ',header=None)
     y_test = pd.read_csv('/content/gdrive/My Drive/HAPT Data Set/Test/y_test.txt',header=None)

     self.lreg.fit(X_train, y_train)
     #self.dtree.fit(X_train, y_train)
     #self.xgb.fit(X_train, y_train)
     #self.ltb.fit(X_train, y_train)
     #self.rf.fit(X_train, y_train)
     if not (path.exists(self.model_filename)):
         model_file = open(self.model_filename, 'w+')
         model_file.close()
     joblib.dump(self.lreg, self.model_filename)

     # Get the score of the model
     prediction = self.lreg.predict_proba(X_test)
     prediction_int = prediction[:, 1] >= 0.75
     prediction_int = prediction_int.astype(np.int)

     cache.set(self.model_cache_key, self.lreg, None)
     cache.set(self.vectorizer_cache_key, self.tfidf_vectorizer, None)
     cache.set(self.score_cache_key, f1_score(y_test, prediction_int), None)
 # Get score
 def get_score(self):
     score = cache.get(self.score_cache_key)
     if score:
         return score
     return 'No score in cache'
 # Predict Human activity
 def predict_single_tweet(self, activity):

     model = cache.get(self.model_cache_key)

     if model is None:
         model_path = path.realpath(self.model_rel_path)
         model = joblib.load(model_path)

         # save in django memory cache

         cache.set(self.model_cache_key, model, None)

     return model.predict(activity)[0]

