# Disaster Response Pipeline Project

### Executive Summary:
This project applies most of the topics learned in the fourth lesson of Udacity Data Science Nano Degree, Data Engineering:
With natural language processing methods and machine learning skills message data, which was sent during natural disasters, was analyzed to classify the messages into 36 categories. This should help response teams to categorize the messages according to urgency.

### File Descriptions:
The repository consists of three main folders:
    data:
        - categories.csv: dataset including all the categories
        - messages.csv: dataset including all messages
        - process_data.py: reads in the data, cleans and stores it in a SQL database
    models:
        - train_classifier.py: includes the code necessary to load data, transform it using natural language processing, run a           machine learning model using GridSearchCV and train it
    app:
        -  Flask app and the user interface used to predict results and display them
        templates: folder containing the html templates
    

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python3 data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python3 models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python3 run.py`

3. Go to http://0.0.0.0:3001/

