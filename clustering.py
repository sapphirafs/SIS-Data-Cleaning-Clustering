
"""
Data cleaning and clustering for an student-information-system (SIS) dataset

The code performs data cleaning and clustering on a dataset whilst addresses missing data, 
cleaning/normailsing column names, and formating text fields. 
Additionaly, it applies K-Means clustering for grouping similar data points.

Proccesses:
1. Replace missing values with the placeholder, "missing_data".
2. Standardize/clean text in columns
3. split overloaded columns into new ones.
4. Normalise text.
5. Run K-Means clustering to group data into 4 groups.
6. Save the cleaned dataset.
"""

# Import necessary libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# Create a label encoder object
encoder = LabelEncoder()

# Load the dataset
df = pd.read_csv(r'C:\Users\PLACEHOLDER\SIS_Faculty-List.csv', encoding='ISO-8859-1')

#Handle missing data - replace missing values with placeholder
# By doing this, missing values wil not interfere with the machine learning model analasys
df.fillna("missing_data", inplace=True)

# Clean column names
# replace "new line" chars and spaces with underscores and convert to lowercase
df.columns = df.columns.str.replace('\n', ' ').str.replace(' ', '_').str.lower()

# remove duplicate rows so the dataset only has unique entries
df.drop_duplicates(inplace=True)

# make sure 'id' column is treated as a string (it could be stored as a number)
df['id'] = df['id'].astype(str)

# normalise text formatting by stripping extra spaces and capitalising names
df['name'] = df['name'].str.strip().str.title()

# Split overloaded columns into new ones
# The 'document_other_professional_certification_critiera' column contains multiple types of data
# So, here, I use regular expressions (based on spacing in the values) to split it into three different columns:
#  'work_experience', 'teaching_excellence', and 'certificates'
#This needs more attention and more specific reg ex,
#   to successfully split the data based on key words.
print(df.columns)  # Checking column names to confirm the right one is selected
df[['work_experience', 'teaching_excellence', 'certificates']] = df[
    'document_other_professional_certification_critiera_five_years_work_experience_teaching_excellence_professional_certifications'].str.extract(
        r'(\s{2,}).*(\s{2,})*(\s{2,})*')
# convert the 'join_date' column to datetime format
#  so dates can be handled correctly by the ML model
df['join_date'] = pd.to_datetime(df['join_date'], errors='coerce')
# standardise and strip text fields in other columns
df['highest_qualification'] = df['highest_qualification'].str.strip().str.title()
df['highest_qualification_level'] = df['highest_qualification_level'].str.strip().str.title()


# # Save the cleaned dataset to a new CSV file
# # this is useful, to use this cleaned data for further analysis or modeling
# NEW_FILE_CREATION = "C:....\\Cleaned5_SIS_Faculty-List.csv"
# os.makedirs(os.path.dirname(NEW_FILE_CREATION), exist_ok=True)
# df.to_csv(NEW_FILE_CREATION, index=False)

#############################
# PART TWO, K-MEANS CLUSTERING
#############################

# identify categorical columns (non-numeric columns)
categorical_columns = df.select_dtypes(include=['object']).columns

# iterate over columns and encode each categorical column using sklearns's LabelEncoder to even out the data into numeric form
# so that the data is approriate for clustering
for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])
# perform clustering on numeric data
#  select columns that have numeric data
num_df = df.select_dtypes(include=[np.number])

# show the columns in the df
print(num_df.head())

# If numeric columns exist, apply K-Means clustering
if not num_df.empty:
    # start the K-Means modeling with 4 clusters (groupings)
    model = KMeans(n_clusters=4, random_state=50)
    
    # fit the model to the new numeric data
    model.fit(num_df)
    
    # add a new column called cluster to the dataframe (excel file) to label the clusters as belonging to 1,2,3 or 4.
    df['cluster'] = model.labels_

    # Print out the centroids of the clusters to understand the center of each group
    print("Centroids of clusters:")
    print(model.cluster_centers_)

    # plot the clusters to visualize the clustering result in an ipynb file in juniper notebook
    plt.figure(figsize=(8, 6))
    
    # Plot the data points (alter the x and y axis to see different corresponding data), 
    #make different clusters diferent colurs with "viridis"
    plt.scatter(num_df.iloc[:, 2], num_df.iloc[:, 5], c=model.labels_, s=50, cmap='viridis')
    
    # plot the centroids of the clusters with red X's 
    plt.scatter(model.cluster_centers_[:, 2], model.cluster_centers_[:, 5], color='red', marker='X', s=200)
    plt.title('K-Means Clustering')
    plt.show()
else:
    print("Numeric data is unavailable for clustering.")

# see the cleaned data by printing
#  information and preview the cleaned dataset
print(df.info())
print(df.head())

# Save the cleaned dataset to a new CSV file
# this is useful, for seeing cleaned data for further analysis or modeling
NEW_FILE_CREATION = "C:\PLACEHOLDER\Cleaned5_SIS_Faculty-List.csv"
os.makedirs(os.path.dirname(NEW_FILE_CREATION), exist_ok=True)
df.to_csv(NEW_FILE_CREATION, index=False)