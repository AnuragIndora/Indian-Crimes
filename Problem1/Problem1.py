import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load your data
data = pd.read_pickle("FeatureCrimeData.pkl")
df = data.copy()

# Impute missing values (if necessary)
imputer = SimpleImputer(strategy='most_frequent')
df['victim_gender'] = imputer.fit_transform(df[['victim_gender']])

# Encode categorical variables
label_encoders = {}
categorical_columns = ['crime_description', 'weapon_used', 'crime_domain', 'broad_crime_domain',
                       'victim_age_group', 'crime_time_interaction', 'age_gender_interaction']

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le  # Store the encoder for inverse_transform if needed

# Feature scaling
scaler = StandardScaler()
numerical_columns = ['victim_age', 'police_deployed', 'time_to_report', 'time_to_close_case',
                     'city_crime_rate', 'crime_description_length', 'police_deployment_density',
                     'city_crime_count', 'avg_police_deployed_domain']

df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Define feature matrix X and target vector y
X = df[['crime_code', 'crime_description', 'weapon_used', 'victim_age', 'victim_gender',
        'police_deployed', 'time_to_report', 'time_to_close_case', 'hour_of_occurrence',
        'day_of_week', 'month_of_occurrence', 'year_of_occurrence', 'city',
        'city_crime_rate', 'crime_description_length', 'broad_crime_domain',
        'victim_age_group', 'police_deployment_density', 'quick_case_closure',
        'crime_time_interaction', 'age_gender_interaction', 'city_crime_count',
        'avg_police_deployed_domain']]

y = df['case_closed']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting Machine": GradientBoostingClassifier(random_state=42),
    "Support Vector Machine": SVC(random_state=42)
}

# Train and evaluate each model
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"--- {model_name} ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\n")

for model_name, model in models.items():
    evaluate_model(model, X_train, X_test, y_train, y_test, model_name)

