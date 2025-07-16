# SIS Data Cleaning & Clustering (Python)

This project cleans and performs K-Means clustering on a Student Information System (SIS) dataset.

## Project Description

The code takes in a raw CSV file containing SIS faculty data and performs the following steps:

1. Replaces missing values with a *missing_data* placeholder.
2. Cleans and standardizes column names.
3. Removes duplicate rows.
4. Normalizes and formats text fields (like names and qualifications).
5. Splits overloaded columns into more usable ones (e.g., splitting certifications).
6. Encodes categorical variables numerically using *LabelEncoder*.
7. Performs K-Means clustering with 4 clusters.
8. Saves the cleaned and labeled dataset as a new CSV file.
9. Visualizes the clusters using Matplotlib.

## Requirements

- Python 3.7+
- Pandas
- NumPy
- scikit-learn
- Matplotlib

Install dependencies using:

```bash
pip install pandas numpy scikit-learn matplotlib
