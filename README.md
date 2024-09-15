# Disaster Response Pipeline Project

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