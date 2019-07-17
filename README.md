# flask-api-to-predict-sentiment-of-reviews

One of the purposes of training a Machine Learning model is to use them in an application or any other products/services which an end user can use. Api's are a way to link a machine learning model to a front end developed in any other language like Java etc. Flask is a micro-web framework used to create REST APIs and performs the required function (prediction) of a ML model by receiving input in the pre-defined format.

This repository contains code for flask api which wraps the neural network models created to predict sentiment of reviews. The api contains the following three models,  
* Bag of Words model (BOW)
* Convolutional Neural Network (CNN) model
* LSTM model

This repository does not contain code for training the models. One of the [other repositories](https://github.com/nithishkaviyan/Sentiment-Analysis-of-Yelp-Reviews) of mine contains detailed code for training. This api uses a pre-trained model and predicts after receiving input posted through its hosted server.

The input must be in json format. A sample input format can be found in sample_request.py file.





Reference:

[1]. https://www.datacamp.com/community/tutorials/machine-learning-models-api-python
