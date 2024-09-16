# Disaster Response Pipeline Project

### Introduction

In this Project, we'll find a data set containing real messages that were sent during disaster events. We will be creating a machine learning pipeline to categorize these events so that we can send the messages to an appropriate disaster relief agency. The objective of the project is to create a web application which will use machine learning pipeline to classify disaster messages.

### File Description

- data:
  - disaster_categories.csv - Contains the id, message that was sent and genre.
  - disaster_messages.csv - Contains the id and the categories (related, offer, medical assistance, etc.) the message belonged to.
  - disaster_response_db.db - Database contain cleaned data
  - process_data.py - Used for data cleaning and pre-processing
- models:
  - train_classifier.py - Used to train the model
  - classifier.pkl - Trained model
- app:
  - run.py - To run Web App
  - templats: HTML templates to show the data

### **Prerequisites**

All the required libraries are included in the file <code>requirements.txt</code>

### Installation

1. Go to Project's root directory
2. Install required libraries.

```
pip install -r requirements.txt
```

3. To run ETL Pipeline that clean and store data

```
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response_db.db
```

4. To run ML pipeline that trains the classifier model

```
python models/train_classifier.py data/disaster_response_db.db models/classifier.pkl
```

5. To displays visualizations that describe the training data run

```
cd app/
python run.py
```

6. Open the web browser and open

```
http://localhost:3001/
```
