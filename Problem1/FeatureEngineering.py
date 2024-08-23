import pandas as pd
import numpy as np

# Load the preprocessed dataset
data = pd.read_pickle("/crime_dataset.pkl")
data.head()
data.info()

# Create a copy of the dataset for feature engineering
df = data.copy()

# Extract features from datetime columns
df['hour_of_occurrence'] = df['time_of_occurrence'].dt.hour
df['day_of_week'] = df['date_of_occurrence'].dt.dayofweek
df['month_of_occurrence'] = df['date_of_occurrence'].dt.month
df['year_of_occurrence'] = df['date_of_occurrence'].dt.year

# Calculate time intervals
df['time_to_report'] = (df['date_reported'] - df['date_of_occurrence']).dt.days
df['time_to_close_case'] = (df['date_case_closed'] - df['date_of_occurrence']).dt.days

# Calculate city crime rate
city_crime_rate = df['city'].value_counts(normalize=True)
df['city_crime_rate'] = df['city'].map(city_crime_rate)

# Encode city as numerical values
df['city'] = df['city'].astype('category').cat.codes

# Calculate the length of the crime description
df['crime_description_length'] = df['crime_description'].apply(len)

# Define a function to group crime domains into broader categories
def broad_grouping_function(crime_domain):
    if crime_domain in ['Violent Crime']:
        return 'Severe Crime'
    elif crime_domain in ['Fire Accident', 'Traffic Fatality']:
        return 'Accidental'
    else:
        return 'Other'

# Apply the function to create a new column for broader crime domains
df['broad_crime_domain'] = df['crime_domain'].map(broad_grouping_function)

# Create age group categories
bins = [0, 12, 18, 60, 100]
labels = ['child', 'teen', 'adult', 'senior']
df['victim_age_group'] = pd.cut(df['victim_age'], bins=bins, labels=labels)

# Encode victim gender as numerical values
df['victim_gender'] = df['victim_gender'].map({'Male': 0, 'Female': 1})

# Calculate police deployment density
avg_police_deployed = df.groupby('crime_code')['police_deployed'].transform('mean')
df['police_deployment_density'] = df['police_deployed'] / avg_police_deployed

# Encode case closed as binary values
df['case_closed'] = df['case_closed'].map({'Yes': 1, 'No': 0})

# Create a binary feature for quick case closure
df['quick_case_closure'] = df['time_to_close_case'].apply(lambda x: 1 if x <= 30 else 0)

# Create interaction features
df['crime_time_interaction'] = df['crime_code'].astype(str) + '_' + df['hour_of_occurrence'].astype(str)
df['age_gender_interaction'] = df['victim_age_group'].astype(str) + '_' + df['victim_gender'].astype(str)

# Calculate city crime count and average police deployment by crime domain
df['city_crime_count'] = df.groupby('city')['city'].transform('count')
df['avg_police_deployed_domain'] = df.groupby('crime_domain')['police_deployed'].transform('mean')

# Save the engineered dataset
df.to_pickle("./FeatureCrimeData.pkl")

# Display the engineered features
df.head()
df.columns
