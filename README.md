# sports-choice-multinomial-logistic-regression
A lightweight Python implementation of a multinomial logistic regression (softmax) classifier predicting a person's preferred sport (Cricket, Football, or Tennis) based on weekly activity hours.

# Sports Preference Predictor using Multinomial Logistic Regression

A simple Python script demonstrating how multinomial logistic regression (softmax classification) works from scratch using NumPy. It calculates log-odds (logits) based on time spent playing different sports and converts them into normalized prediction probabilities.

## Features
* Calculates log-odds for multiple classes relative to a baseline category (Cricket).
* Applies the **Softmax function** to convert logits into valid probability distributions.
* Predicts the most likely preferred sport based on input hours.

## Requirements
* Python 3.x
* NumPy

## Usage
Run the script directly:
```bash
python Games.py
