# Netflix Stock Price Prediction

This repository contains code for predicting Netflix stock prices using an LSTM (Long Short-Term Memory) model. The code is written in Python and utilizes libraries such as pandas, numpy, scikit-learn, and TensorFlow.

## Dataset
The dataset used for training and testing the model is the Netflix stock price data. The data is provided in a CSV file, which includes historical stock prices including the 'Close' price column. Download the Netflix stock price dataset from [here](https://www.kaggle.com/datasets/jainilcoder/netflix-stock-price-prediction).


## Model Architecture
The LSTM model is implemented using the Keras API from TensorFlow. The model consists of an LSTM layer with 64 units followed by a Dense layer with 1 unit. The model is compiled with the Adam optimizer and mean squared error (MSE) loss function.

## Data Preprocessing
The 'Close' price column is extracted from the dataset and scaled using the MinMaxScaler from scikit-learn. The data is then divided into training and testing sets. Input sequences of a specified length `sequence_length` are created along with their corresponding target values.

## Training and Evaluation
The model is trained on the training data using the created sequences and target values. The training is performed for a specified number of epochs with a batch size of 64. After training, the model is evaluated on the test data to measure its performance using the mean squared error (MSE) loss.

## Prediction
Using the trained model, predictions are made on the test data. The predicted values are then inverse scaled to obtain the actual stock prices. The predicted and actual prices are printed to compare the model's performance.

## Requirements
- pandas
- numpy
- scikit-learn
- TensorFlow

The dependencies can be installed using the following command:
````
pip install -r requirements.txt
````
