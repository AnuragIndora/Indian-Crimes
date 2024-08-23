# Indian-Crimes

Here's the updated `README.md` file with the column descriptions and the Kaggle link included:

---

# Crime Dataset India - Data Preprocessing and Exploratory Data Analysis

This repository contains the code and analysis performed on the `crime_dataset_india.csv` file, focusing on data preprocessing and exploratory data analysis (EDA) to uncover insights and patterns in crime data from India.

## Table of Contents

- [Overview](#overview)
- [Dataset Description](#dataset-description)
  - [Columns Description](#columns-description)
  - [Kaggle Dataset](#kaggle-dataset)
- [Data Preprocessing](#data-preprocessing)
  - [Task 1: Handling Different Date Formats](#task-1-handling-different-date-formats)
  - [Task 2: Converting Object Columns to Datetime](#task-2-converting-object-columns-to-datetime)
  - [Task 3: Handling Missing Values](#task-3-handling-missing-values)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Dependencies](#dependencies)
- [Usage](#usage)

## Overview

This project is aimed at:

1. **Data Preprocessing:** Cleaning and transforming the dataset to prepare it for analysis.
2. **Exploratory Data Analysis (EDA):** Exploring the dataset to understand the relationships between different features and uncover patterns in crime data.

## Dataset Description

The dataset used for this project provides detailed records of various crimes reported in India. It is publicly available on Kaggle and contains multiple columns that describe different aspects of each crime incident.

### Columns Description

- **date_reported:** The date and time when the crime was reported.
- **date_of_occurrence:** The specific date when the crime occurred.
- **time_of_occurrence:** The specific time when the crime occurred.
- **city:** The city where the crime was committed.
- **crime_code:** A code that categorizes the type of crime.
- **crime_description:** A detailed description of the crime committed.
- **victim_age:** The age of the victim involved in the crime.
- **victim_gender:** The gender of the victim involved in the crime.
- **weapon_used:** The type of weapon used in the crime.
- **crime_domain:** The broader category or domain of the crime (e.g., violent crime, property crime).
- **police_deployed:** The number of police officers deployed to handle the case.
- **case_closed:** Indicates whether the case was closed ("Yes") or remains open ("No").
- **date_case_closed:** The date and time when the case was officially closed.

### Kaggle Dataset

The dataset used in this analysis can be found on Kaggle:

[Indian Crimes Dataset on Kaggle](https://www.kaggle.com/datasets/sudhanvahg/indian-crimes-dataset)

## Data Preprocessing

The preprocessing steps involve several tasks to clean and structure the data effectively.

### Task 1: Handling Different Date Formats

In the `crime_dataset_india.csv` file, date columns contain data in multiple formats, such as `"%d/%m/%Y %H:%M"` and `"%d-%m-%Y %H:%M"`. These variations cause issues when parsing dates.

**Solution:**
- We utilized a function to detect and parse dates in both formats, converting them into a unified datetime format. This approach minimizes missing values that would arise from trying to parse the dates using a single format.

### Task 2: Converting Object Columns to Datetime

Several columns in the dataset contain date information but are stored as object types.

**Solution:**
- After handling the different date formats, these columns were converted into a proper datetime format, ensuring that they can be used effectively in time-series analysis and other date-related operations.

### Task 3: Handling Missing Values

The dataset had missing values in critical columns like `weapon_used` and `date_case_closed`.

**Solution:**
- **Weapon Used:** Missing values in `weapon_used` were imputed a specific weapon names 'weapon'.
- **Date Case Closed:** Missing dates in the `date_case_closed` column were filled using unknown domain.

### Task 4: Save Processed Data in Pickle Format

After completing the preprocessing steps, the processed dataset is saved in a more efficient and portable binary format using Python's pickle module. This allows for quick loading in future analyses.

- **Output:** The preprocessed dataset is saved as `crime_dataset.pkl`.

This pickle file can be easily loaded for subsequent steps in the analysis or for use in machine learning models.

## Exploratory Data Analysis (EDA)

The EDA focused on understanding the relationships between different features in the dataset:

- **Analyzing Individual Columns:**
  - Distribution and frequency analysis of crime types, locations, weapon usage, and other features.
  - Time series analysis to explore trends in crime reporting and resolution over time.

- **Correlation Analysis:**
  - Examining the correlation between numerical features, such as police deployment, crime frequency, case closure rates, and more.
  - Visualizing these correlations using heatmaps to identify strong relationships between variables.

Here’s the updated portion with the new dependencies and instructions:

---

## Dependencies

The following Python libraries are required to run the analysis:

- `pandas` - For data manipulation and analysis.
- `numpy` - For numerical computing and handling arrays.
- `matplotlib` - For creating static, animated, and interactive visualizations.
- `seaborn` - For making statistical graphics in Python.
- `datetime` - For manipulating dates and times.
- `matplotlib.dates` - For working with dates in matplotlib plots.
- `plotly.express` - For interactive data visualizations.
- `wordcloud` - For generating word clouds.
- `sklearn.feature_extraction.text` - For converting a collection of text documents to a matrix of token counts.
- `warnings` - For issuing warnings (used to suppress FutureWarnings in this analysis).

Install these dependencies using pip:

```bash
pip install pandas numpy matplotlib seaborn plotly wordcloud scikit-learn
```

To ensure a smooth experience, it’s recommended to suppress FutureWarnings, which can clutter the output. This can be done using the following code in your script:

```python
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
```

These libraries provide the foundation for performing data preprocessing, exploratory data analysis, and visualizations.

Here’s the updated **Usage** section:

---

## Usage

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/crime-dataset-india-analysis.git
   cd crime-dataset-india-analysis
   ```

2. **Run the Preprocessing and EDA Notebooks:**

   - Ensure the `crime_dataset_india.csv` file is in the project directory.
   - Open and execute the notebooks in Jupyter:

   ```bash
   jupyter notebook Preprocessing.ipynb
   jupyter notebook ExporatoryDataAnalysis.ipynb
   ```

3. **Review the Output:**

   - The notebooks will generate several visualizations and a summary report of the EDA.
   - Check the output cells and any saved output files for detailed insights.

---
