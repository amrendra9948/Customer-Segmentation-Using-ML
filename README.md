# Customer-Segmentation-Using-ML
This project uses Machine Learning to divide customers into different groups based on their annual income and spending behaviour.

# Customer Segmentation Using Machine Learning

## Project Overview

This project uses Machine Learning to divide customers
into different groups based on their annual income and
spending behaviour.

The K-Means Clustering algorithm is used to identify
similar customer groups.

An interactive Streamlit dashboard is developed to
visualize customer segments and predict the group
of a new customer.

## Objectives

- Understand customer shopping behaviour.
- Group customers using K-Means Clustering.
- Identify high-income and high-spending customers.
- Visualize different customer segments.
- Predict the cluster of a new customer.

## Dataset

Dataset: Mall Customer Segmentation Dataset

Source:
https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

Dataset contains:
- CustomerID
- Gender
- Age
- Annual Income (k$)
- Spending Score (1-100)

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Jupyter Notebook
- Joblib

## Machine Learning Algorithm

K-Means Clustering (Unsupervised Learning)

## Project Workflow

1. Dataset Collection
2. Data Cleaning
3. Exploratory Data Analysis
4. Feature Selection
5. Feature Scaling
6. Elbow Method
7. K-Means Model Training
8. Silhouette Score Evaluation
9. Customer Segmentation
10. Streamlit Dashboard Development

## Project Structure

Customer_Segmentation_ML/
|
|-- data/
|   |-- Mall_Customers.csv
|
|-- notebooks/
|   |-- Customer_Segmentation.ipynb
|
|-- models/
|   |-- customer_segmentation_model.pkl
|
|-- outputs/
|   |-- clustered_customers.csv
|
|-- app.py
|-- requirements.txt
|-- README.md
|-- .gitignore

## Installation

Clone the repository:

git clone YOUR_GITHUB_REPOSITORY_URL

Open the project:

cd Customer_Segmentation_ML

Create a virtual environment:

python -m venv venv

Activate it on Windows:

venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

## Run the Project

First, open the Jupyter Notebook and run all cells
to train and save the ML model.

Then start the Streamlit application:

streamlit run app.py

Open:

http://localhost:8501

## Model Evaluation

The model is evaluated using:

- Elbow Method
- Silhouette Score
- Inertia
- Cluster Distribution

Actual evaluation results should be added
after executing the notebook.

## Application Features

- Interactive customer segmentation dashboard
- Customer cluster visualization
- Customer segment summary
- New customer cluster prediction
- Downloadable segmentation results

## Future Improvements

- Use larger real-world retail datasets.
- Add customer transaction history.
- Implement RFM analysis.
- Compare K-Means with DBSCAN and
  Hierarchical Clustering.
- Deploy the application online.

## Author

Amrendra Bahadur Singh

Machine Learning Project
