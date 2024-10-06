import pandas as pd

# Load the Excel file
file_path = 'D:/JOP_TRACKS/DataScirnceAndAnalysis/IBM Data Scientist Materials/HeartProject/T.xlsx'

# Option 1: Using pd.ExcelFile to access sheet names
xls = pd.ExcelFile(file_path)  # Load the Excel file
sheet_names = xls.sheet_names  # Get the names of the sheets
#print(sheet_names)

# Option 2: Using pd.read_excel to read all sheets
df = pd.read_excel(file_path, sheet_name=None)
# print(df)

# Iterate over the sheets and load them into a DataFrame
for sheet in sheet_names:
    df = xls.parse(sheet)
    #print(f"\nPreview of sheet: {sheet}")
    #print(df.head())  # Show the first few rows of each sheet


# Check for missing data in each sheet
for sheet in sheet_names:
    df = xls.parse(sheet)
    print(f"\nMissing data in sheet: {sheet}")
    print(df.isnull().sum())

# Dealing With Each Table

import pandas as pd

# Load the Excel file
file_path = 'D:/JOP_TRACKS/DataScirnceAndAnalysis/IBM Data Scientist Materials/HeartProject/T.xlsx'

# Load all sheets into a dictionary of DataFrames
sheets_dict = pd.read_excel(file_path, sheet_name=None)

# Print the sheet names
print(sheets_dict.keys())  # This will show all the sheet names


# Load the 'blood' sheet
df_blood = sheets_dict['Blood']

# Preview the data
print(df_blood.head())

# Load the 'blood_ins' sheet
df_blood_ins = sheets_dict['Blood_Ins']

# Preview the  data
print(df_blood_ins.head())

# Load the 'Patient_Info' sheet
df_Patient_Info = sheets_dict['Patient_Info']

# Preview the data
print(df_Patient_Info.head())

# Load the 'More_Info' sheet
df_More_Info = sheets_dict['More_Info']

# Preview the data
print(df_More_Info.head())


# Load the 'Sleepness' sheet
df_Sleepness = sheets_dict['Sleepness']

# Preview the data
print(df_Sleepness.head())



# Load the 'Drunking' sheet
df_Drunking = sheets_dict['Drunking']

# Preview the data
print(df_Drunking.head())

# Explore of blood table and 
# Check the shape of the dataset
print(df_blood.shape)

# Check the data types of the columns
print(df_blood.dtypes)

# Check for missing values
print(df_blood.isnull().sum())

# Get summary statistics of the numerical columns
print(df_blood.describe())

# Describe blood_ins table

# Check the shape of the dataset
print(df_blood_ins.shape)

# Check the data types of the columns
print(df_blood_ins.dtypes)

# Check for missing values
print(df_blood_ins.isnull().sum())

# Get summary statistics of the numerical columns
print(df_blood_ins.describe())

# Visualize Mean Important Columns


import matplotlib.pyplot as plt
import seaborn as sns

# Plot the distribution of a specific column
sns.histplot(df_blood_ins['blood_sugar'], kde=True)
plt.title('Distribution of blood_sugar')
plt.show()


# Calculate correlation matrix
correlation_matrix = df_blood_ins.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# Plot the distribution of a specific column
sns.histplot(df_blood['Pd'], kde=True)
plt.title('pd')
plt.show()

# Plot the distribution of a specific column
sns.histplot(df_blood['Ps'], kde=True)
plt.title('Ps')
plt.show()


#Step 3 Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# cleaning and drop unwanted coulnms

# Select features and target variable
# Assuming you want to predict 'blood_pressure', 'insulin', 'sleepness', and 'drunkenness'
# Adjust the target variable and features according to your dataset
features = df_blood_ins[['PTT', 'HR', 'SpO2']]  # Replace with actual feature names
target = df_blood_ins[['blood_sugar', 'insulin']]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Initialize the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')



from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)

# Best parameters
print(f'Best Parameters: {grid_search.best_params_}')

# Best estimator
best_model = grid_search.best_estimator_

# Evaluate the optimized model
y_pred_optimized = best_model.predict(X_test_scaled)
mse_optimized = mean_squared_error(y_test, y_pred_optimized)
r2_optimized = r2_score(y_test, y_pred_optimized)

print(f'Optimized Mean Squared Error: {mse_optimized}')
print(f'Optimized R^2 Score: {r2_optimized}')

#-------------------------------------------------------
# cleaning and drop unwanted coulnms

# Select features and target variable
# Assuming you want to predict 'blood_pressure', 'insulin', 'sleepness', and 'drunkenness'
# Adjust the target variable and features according to your dataset
blood_features = df_blood[['PTT', 'HR', 'SPO2']]  # Replace with actual feature names
blood_target = df_blood[['Pd', 'Ps']]

# Split the data
blood_X_train, blood_X_test, blood_y_train, blood_y_test = train_test_split(blood_features, blood_target, test_size=0.2, random_state=42)


# Scale the features
scaler = StandardScaler()
blood_X_train_scaled = scaler.fit_transform(blood_X_train)
blood_X_test_scaled = scaler.transform(blood_X_test)


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Initialize the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(blood_X_train_scaled,blood_y_train)

# Make predictions
blood_y_pred = model.predict(blood_X_test_scaled)

# Evaluate the model
blood_mse = mean_squared_error(y_test, y_pred)
blood_r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error blood : {blood_mse}')
print(f'R^2 Score blood: {blood_r2}')



#classification of the outpout of suger and insuline 
# Define thresholds for blood sugar, insulin, etc. (example ranges)
def classify_blood_sugar(value):
    if value < 70:
        return 'Danger'
    elif 70 <= value < 100:
        return 'Semi-normal'
    else:
        return 'Normal'

# Apply classification to your data
df_blood_ins['blood_sugar_state'] = df_blood_ins['blood_sugar'].apply(classify_blood_sugar)
df_blood_ins['insulin_state'] = df_blood_ins['insulin'].apply(lambda x: 'Danger' if x > 200 else 'Normal')

# Preview results
print(df_blood_ins[['blood_sugar', 'blood_sugar_state', 'insulin', 'insulin_state']].head())


from sklearn.cluster import KMeans

# Assuming we want to cluster based on blood sugar and insulin levels
X = df_blood_ins[['blood_sugar', 'insulin']]

# Initialize KMeans with 3 clusters for "Normal", "Semi-normal", and "Danger"
kmeans = KMeans(n_clusters=3, random_state=42)
df_blood_ins['state_cluster'] = kmeans.fit_predict(X)

# Map the clusters to labels based on the analysis of centroids
df_blood_ins['state'] = df_blood_ins['state_cluster'].map({0: 'Normal', 1: 'Semi-normal', 2: 'Danger'})

# Preview the results
print(df_blood_ins[['blood_sugar', 'insulin', 'state']].head())


import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Load your trained model and scaler
# Here you need to ensure that 'model' and 'scaler' are defined and trained as shown in your ML section.
# For example:
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# scaler = StandardScaler()
# Load them from disk if saved using joblib or pickle.

# Function to predict blood sugar and insulin
def predict_blood_levels(PTT, HR, SPO2):
    # Scale the input features
    input_data = scaler.transform([[PTT, HR, SPO2]])
    
    # Make predictions
    predictions = model.predict(input_data)
    
    blood_sugar, insulin = predictions[0]
    return blood_sugar, insulin

# Streamlit UI
st.title('Blood Sugar and Insulin Prediction')

PTT = st.number_input('PTT', min_value=0.0)
HR = st.number_input('Heart Rate (HR)', min_value=0.0)
SPO2 = st.number_input('Oxygen Saturation (SpO2)', min_value=0.0)

if st.button('Predict'):
    blood_sugar, insulin = predict_blood_levels(PTT, HR, SPO2)
    
    # Classify the results
    def classify_blood_sugar(value):
        if value < 70:
            return 'Danger'
        elif 70 <= value < 100:
            return 'Semi-normal'
        else:
            return 'Normal'

    blood_sugar_state = classify_blood_sugar(blood_sugar)
    insulin_state = 'Danger' if insulin > 200 else 'Normal'
    
    # Display the results
    st.write(f'Predicted Blood Sugar: {blood_sugar} - State: {blood_sugar_state}')
    st.write(f'Predicted Insulin: {insulin} - State: {insulin_state}')
