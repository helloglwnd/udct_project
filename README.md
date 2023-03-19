
# Disaster Response Model
This project focuses on building a machine learning model to classify messages received during a disaster. The data for this project was provided by Figure Eight and contains real messages that were sent during disaster events. The model is built using a scikit-learn pipeline, GridSearchCV, and Random Forest Classifier.



# Requirements
* Python 3.5+
* Numpy
* Pandas
* Scikit-learn
* NLTK
* SQLalchemy
* Pickle

# Files
* process_data.py: Script to clean and save data into SQLite database
* train_classifier.py: Script to build, train, and evaluate model
* data/disaster_categories.csv: File containing message categories
* data/disaster_messages.csv: File containing messages
* models/classifier.pkl: Saved model after training

# Usage
1. Run the following command in the project's root directory to set up your database and model.

    `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

2. Run the following command in the project's root directory to train your model.

    `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Use the model to make predictions on new data.
