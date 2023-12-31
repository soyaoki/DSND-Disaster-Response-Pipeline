# Disaster Response Pipeline Project

# 1. Project Overview

This repository contains a Flask web app for disaster message classification, utilizing data from [Appen](https://appen.com/). The project involves building a machine learning pipeline to categorize messages, aiding emergency workers in directing information to the appropriate relief agencies. The web app allows users to input new messages and receive classification results across various categories, accompanied by visualizations for enhanced data interpretation.

__Key Features__:
- Flask web app for disaster message classification.
- Machine learning pipeline for message categorization.
- User-friendly interface for emergency workers to input messages and obtain classification results.
- Visualizations to provide insights into the data.

Below are a few screenshots of the web app.

## __Main Page__
![](/main-page.png)
## __Message Classification Result Page__
![](/classification-result-page.png)

# 2. Files in the repository

- `Jupyter-Notebook`
    - `ETL Pipeline Preparation.ipynb` : A notebook for designing and testing ETL steps. Origin of 'process_data.py'.
    - `ML Pipeline Preparation.ipynb` : A notebook for designing and testing ML steps. Original of 'train_classifier.py'.
- `app`
    - `templates`
        -  `master.html`  : HTML file of main page of the web app.
        -  `go.html` : HTML file of classification result page of the web app.
    - `run.py` : A python script for Flask web app using SQLite data and a pre-trained classifier for message classification.
- `data`
    - `disaster_categories.csv` : Data for training included 36 categories.
    - `disaster_messages.csv` : Data for training included text messages.
    - `process_data.py` : A python script to prepare clean data for ML. It's used at ETL(Extract, Transfor, Load) step.
- `models`
    - `train_classifier.py` : A python script to train a message classifier of 36 categories. It's used at ML(Machine Learning) step.

# 3. Instructions:
1. Run the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click [here](http://0.0.0.0:3000/) to open the homepage. If you're working at Udacity's workspace, click the `PREVIEW` button to open the homepage.

# 4. Libraries used

`numpy`, `pandas`, `sklearn`, etc.

For more detail, see [requirements.txt](/requirements.txt)
