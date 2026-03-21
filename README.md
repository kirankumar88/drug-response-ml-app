# Drug Response Prediction using Machine Learning

## Overview

This project is a machine learning application that predicts whether a patient will respond to a drug based on clinical parameters such as blood glucose level, heart rate, liver toxicity index, drug dosage, and blood pressure.
The project includes exploratory data analysis, model training, evaluation, and deployment using a Streamlit web application.

---
# Live Application

The Streamlit web application for this project is available at:
Streamlit App:
https://drug-response-ml-app-5fgdrhtuf2gxpvwwczfkf6.streamlit.app/

You can use the web application to:

Upload CSV files for batch prediction
Enter patient data manually
Download prediction results
View prediction probabilities

## Project Objectives

* Perform exploratory data analysis (EDA)
* Build a classification model for drug response prediction
* Optimize model performance
* Save trained model as a pipeline
* Deploy model using Streamlit
* Allow predictions using CSV upload or manual input

---

## Machine Learning Model

**Algorithm:** Support Vector Machine (SVM)
**Kernel:** RBF
**Pipeline:** StandardScaler + SVM
**Accuracy:** 0.78
**AUC Score:** 0.86

---

## Features Used

The model uses the following clinical parameters:

* Blood Glucose Level (mg/dL)
* Drug Dosage (mg)
* Heart Rate (BPM)
* Liver Toxicity Index (U/L)
* Systolic Blood Pressure (mmHg)

---

## Project Structure

```
Drug-Response-ML-App/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── data.csv
│   └── sample.csv
│
├── images/
│   ├── boxplot.jpeg
│   ├── correlation_heatmap.jpeg
│   ├── count_plot.jpeg
│   ├── histogram_grid_v2.jpeg
│   ├── pairplot.jpeg
│   └── scatter_plot.png
│
├── model/
│   ├── features.pkl
│   └── pipeline.pkl
│
└── notebook/
    └── drug_response_classification.ipynb
```

---

## Exploratory Data Analysis

The following analyses were performed:

* Distribution plots
* Correlation heatmap
* Pair plots
* Box plots
* Count plots

These visualizations are available in the **images/** folder.

---

## How to Run the Project Locally

### 1. Clone the repository

```
git clone https://github.com/kirankumar88/drug-response-ml-app.git
cd drug-response-ml-app
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run Streamlit app

```
streamlit run app.py
```

---

## Streamlit App Features

* Upload CSV file for batch prediction
* Manual input for single prediction
* Download prediction results
* Sample CSV download
* Prediction probability output

---

## Sample Input Format

| Blood Glucose Level (mg/dL) | Drug Dosage (mg) | Heart Rate (BPM) | Liver Toxicity Index (U/L) | Systolic Blood Pressure (mmHg) |
| --------------------------- | ---------------- | ---------------- | -------------------------- | ------------------------------ |
| 100                         | 20               | 80               | 30                         | 120                            |

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Matplotlib
* Seaborn

---

## Future Improvements

* Add more clinical features
* Use ensemble models
* Deploy on cloud
* Add database integration
* Build REST API
* Improve model accuracy

---

## Author

Kiran Kumar
Machine Learning | Bioinformatics | AI in Healthcare

---

## License

This project is for educational and research purposes.
