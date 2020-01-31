from django.shortcuts import render

# Create your views here.
import models.py.classifier as classifier


classifier = classifier()

# Training the model
classifier.train_model()


# Get score
classifier.get_score()


# Predict Human activity
y_hat = list(y_test.values)
classifier.predict_single_tweet(y_hat[0])