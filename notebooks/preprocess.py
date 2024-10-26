import os
import pandas as pd

DATASETURL = 'https://data.montgomerycountymd.gov/api/views/v76h-r7br/rows.csv'

# Load the data
sales = pd.read_csv(DATASETURL)

# Drop Item Description, Supplier, and Item Type
data = sales.drop(['ITEM DESCRIPTION', 'SUPPLIER', 'ITEM TYPE'], axis=1, errors='ignore') 

# Classify into 3 categories: Low (0), Medium (1), High (2)
data['DEMAND CLASS'] = pd.qcut(data['RETAIL SALES'], q=3, labels=[0, 1, 2])

# Remove non-numeric, non-integer data because it contains string data such as WC and BC
data = data[~data['ITEM CODE'].isin(['WC', 'BC'])]
data['ITEM CODE'] = pd.to_numeric(data['ITEM CODE'], errors='coerce')
data = data.dropna(subset=['ITEM CODE'])
data['ITEM CODE'] = data['ITEM CODE'].astype(int)

# Drop Retail Sales since it's replaced by demand class
data = data.drop(['RETAIL SALES'], axis=1, errors='ignore') 

# Fill any missing numeric data
data.fillna(0, inplace=True)

# Save the preprocessed data
directory = 'datasets'

# Save the preprocessed data to datasets directory
if not os.path.exists(directory):
    os.makedirs(directory)

# Save the dataset as preprocessed.csv into datasets directory
filepath = os.path.join(directory, 'preprocessed1.csv')

# Convert sales dataframe to csv dataset
data.to_csv(filepath, index=False)

if os.path.exists(filepath):
    print('Did not save.')
