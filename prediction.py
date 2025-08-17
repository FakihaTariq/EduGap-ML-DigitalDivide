import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
# Accessing Dataset
data = pd.read_csv("digital_literacy_dataset.csv")

# Data Preprocessing
# Checking for Null/Empty Values
# print(data.isna().sum())
# Found out that everywhere it says "None" it's putting it as a null value so filling it
data.Education_Level.fillna("No School", inplace=True)
# Check again
# print(data.isna().sum()) #0 null values everywhere


# Describe to find Average Marker
print(data['Basic_Computer_Knowledge_Score'].describe())  # Average: 25
print(data['Internet_Usage_Score'].describe())  # Average: 25
print(data['Mobile_Literacy_Score'].describe())  # Average: 26

print()  # Average: 25
print(np.median(data['Internet_Usage_Score']))  # Average: 25
print(np.median(data['Mobile_Literacy_Score']))  # Average: 26


# Cleaning and Mapping
def clean(column):
    data[column] = data[column].astype(str).str.strip().str.title()


clean("Education_Level")
clean("Household_Income")
clean("Location_Type")
clean("Employment_Status")

bins = [0, 24, 40, 60, np.inf]
labels = ['Youth', 'Early Career', 'Midlife', 'Senior']
data['Age_Group'] = pd.cut(data['Age'], bins=bins, labels=labels)


def map(column, mapping):
    data[column] = data[column].map(mapping)


education_map = {'No School': 0, 'Primary': 1, 'Secondary': 2, 'High School': 3}
income_map = {'Low': 0, 'Medium': 1, 'High': 2}
employment_map = {'Farmer': 0, 'Unemployed': 1, 'Student': 2, 'Other': 3, 'Self-Employed': 4}
age_map = {'Youth': 0, 'Senior': 1, 'Early Career': 2, 'Midlife': 3}

map("Education_Level", education_map)
map("Household_Income", income_map)
map("Employment_Status", employment_map)
map("Age_Group", age_map)

# Categorize and Labels
def categorize_1(val):  # For computer skills and internet usage
    if val <= data['Basic_Computer_Knowledge_Score'].quantile(0.63):  # the median value
        return 0  # Below Average
    else:
        return 1 # Above Average


def categorize_2(val):  # For mobile skills
    if val <= data['Internet_Usage_Score'].quantile(0.56):
        return 0  # Below Average
    else:
        return 1  # Above Average


def categorize_3(val):  # For mobile skills
    if val <= data['Mobile_Literacy_Score'].quantile(0.69):
        return 0  # Below Average
    else:
        return 1  # Above Average


data['Computer_Skill_Label'] = data['Basic_Computer_Knowledge_Score'].apply(categorize_1)
data['Internet_Usage_Label'] = data['Internet_Usage_Score'].apply(categorize_2)
data['Mobile_Literacy_Label'] = data['Mobile_Literacy_Score'].apply(categorize_3)


x = data[["Education_Level", "Household_Income", "Employment_Status", "Age_Group"]]
y = data[['Computer_Skill_Label', "Internet_Usage_Label", "Mobile_Literacy_Label"]]

features = data[["Education_Level", "Household_Income", "Employment_Status", "Age_Group"]]
x_df = pd.DataFrame(x, columns=features.columns)
vif_data = pd.DataFrame()
vif_data["Feature"] = x_df.columns
vif_data["VIF"] = [variance_inflation_factor(x_df.values, i) for i in range(x_df.shape[1])]

print(vif_data)


# Splitting, Training, Model

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

base_classifier = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=4,
    min_samples_split=5,
    class_weight='balanced',
    max_features='sqrt',
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)


model = MultiOutputClassifier(base_classifier)

model.fit(xtrain, ytrain)
ypred = model.predict(xtest)


# Convert predictions to DataFrame for easier inspection

ypred_df = pd.DataFrame(ypred, columns=y.columns)

# Loop through each target and print classification metrics
for label in y.columns:
    print(f"\n=== Classification Metrics for: {label} ===")
    print("\nClassification Report:")
    print(classification_report(ytest[label], ypred_df[label]))
    print("Accuracy: ", accuracy_score(ytest[label], ypred_df[label]))

for i, label in enumerate(y.columns):
    print(f"\n Feature importances for: {label}")
    estimator = model.estimators_[i]

    importances = estimator.feature_importances_
    for feature, score in zip(x.columns, importances):
        print(f"{feature:<25} {score:.4f}")

# Feature Importance
features = list(x)
# plt.figure(figsize=(8, 6))
# plt.barh(features, importances, color='skyblue')
# plt.xlabel('Importance')
# plt.title('Feature Importance')
# plt.yticks(rotation=35, size = 7)
# plt.gca().invert_yaxis()
# plt.show()

# MODEL 2
# Creating columns
data['Computer_Gain'] = data['Post_Training_Basic_Computer_Knowledge_Score'] - data['Basic_Computer_Knowledge_Score']
data['Internet_Gain'] = data['Post_Training_Internet_Usage_Score'] - data['Internet_Usage_Score']
data['Mobile_Gain'] = data['Post_Training_Mobile_Literacy_Score'] - data['Mobile_Literacy_Score']

print(data[['Computer_Gain', 'Internet_Gain', 'Mobile_Gain']].describe())
print(np.median(data['Computer_Gain']))
print(np.median(data['Internet_Gain']))
print(np.median(data['Mobile_Gain']))

# Cleaning and Mapping
clean("Engagement_Level")

engagement_map = {'Low': 0, 'Medium': 1, 'High': 2}

map("Engagement_Level", engagement_map)


# Categorize and Labels
def categorize_1(val):  # For computer skills and internet usage
    if val <= data['Computer_Gain'].quantile(.56):
        return 0  # Below Average
    else:
        return 1


def categorize_2(val):  # For mobile skills
    if val <= data['Internet_Gain'].quantile(.55) :
        return 0  # Below Average
    else:
        return 1  # Above Average


def categorize_3(val):  # For mobile skills
    if val <= data['Mobile_Gain'].quantile(.6):
        return 0  # Below Average
    else:
        return 1  # Above Average


data['Computer_Gain_Label'] = data['Computer_Gain'].apply(categorize_1)
data['Internet_Gain_Label'] = data['Internet_Gain'].apply(categorize_2)
data['Mobile_Gain_Label'] = data['Mobile_Gain'].apply(categorize_3)

x = data[
    ["Education_Level", "Household_Income", "Employment_Status", "Age_Group", 'Modules_Completed', 'Engagement_Level',
     'Session_Count', 'Average_Time_Per_Module']]
y = data[['Computer_Gain_Label', "Internet_Gain_Label", "Mobile_Gain_Label"]]

for col in y.columns:
    print(f"\n{col} label distribution:")
    print(y[col].value_counts(normalize=True).round(3))

# Splitting, Training, Model
# print(xtrain.isna().sum())

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

base_classifier = RandomForestClassifier(
    n_estimators=50,
    max_depth=5,
    min_samples_leaf=4,
    min_samples_split=10,
    max_features=0.5,
    bootstrap=True,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1

)

model = MultiOutputClassifier(base_classifier)

model.fit(xtrain, ytrain)
ypred = model.predict(xtest)

# Convert predictions to DataFrame for easier inspection

ypred_df = pd.DataFrame(ypred, columns=y.columns)

# Loop through each target and print classification metrics
for label in y.columns:
    print(f"\n=== Classification Metrics for: {label} ===")
    print("\nClassification Report:")
    print(classification_report(ytest[label], ypred_df[label]))
    print("Accuracy: ", accuracy_score(ytest[label], ypred_df[label]))

features = []
for i, label in enumerate(y.columns):
    print(f"\n Feature importances for: {label}")
    estimator = model.estimators_[i]

    importances = estimator.feature_importances_
    for feature, score in zip(x.columns, importances):
        print(f"{feature:<25} {score:.4f}")
        features.append(feature)

# Feature Importance
# features = list(x)
# plt.figure(figsize=(8, 6))
# plt.barh(features, importances, color='skyblue')
# plt.xlabel('Importance')
# plt.title('Feature Importance')
# plt.yticks(rotation=35, size = 7)
# plt.gca().invert_yaxis()
# plt.show()


