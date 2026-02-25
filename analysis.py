import warnings
warnings.filterwarnings('ignore')
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
print(data.isna().sum()) #0 null values everywhere


# Take a look at the data
print(data['Basic_Computer_Knowledge_Score'].describe())  
print(data['Internet_Usage_Score'].describe())  
print(data['Mobile_Literacy_Score'].describe())  

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
    if val < np.median(data['Basic_Computer_Knowledge_Score']):  # the median value
        return 0  # Below Average
    else:
        return 1 # Above Average

def categorize_2(val):  # For mobile skills
    if val < np.median(data['Internet_Usage_Score']):
        return 0  # Below Average
    else:
        return 1  # Above Average

def categorize_3(val):  # For mobile skills
    if val < np.median(data['Mobile_Literacy_Score']):
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

# Feature Importance
for i, label in enumerate(y.columns):
    print(f"\n Feature importances for: {label}")
    estimator = model.estimators_[i]

    importances = estimator.feature_importances_
    for feature, score in zip(x.columns, importances):
        print(f"{feature:<25} {score:.4f}")

features = list(x)
plt.figure(figsize=(8, 6))
plt.barh(features, importances, color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.yticks(rotation=35, size = 7)
plt.gca().invert_yaxis()
plt.show()

# MODEL 2
# Creating columns
data['Computer_Gain'] = data['Post_Training_Basic_Computer_Knowledge_Score'] - data['Basic_Computer_Knowledge_Score']
data['Internet_Gain'] = data['Post_Training_Internet_Usage_Score'] - data['Internet_Usage_Score']
data['Mobile_Gain'] = data['Post_Training_Mobile_Literacy_Score'] - data['Mobile_Literacy_Score']

print(data[['Computer_Gain', 'Internet_Gain', 'Mobile_Gain']].describe())

# Cleaning and Mapping
clean("Engagement_Level")

engagement_map = {'Low': 0, 'Medium': 1, 'High': 2}

map("Engagement_Level", engagement_map)

# Categorize and Labels
def categorize_1(val):  # For computer skills and internet usage
    if val < np.median(data['Computer_Gain']):
        return 0  # Below Average
    else:
        return 1

def categorize_2(val):  # For mobile skills
    if val < np.median(data['Internet_Gain']):
        return 0  # Below Average
    else:
        return 1  # Above Average

def categorize_3(val):  # For mobile skills
    if val < np.median(data['Mobile_Gain']):
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
ypred_gain_df = pd.DataFrame(ypred, columns=y.columns)

#Feature Importance
features = []
for i, label in enumerate(y.columns):
    print(f"\n Feature importances for: {label}")
    estimator = model.estimators_[i]

    importances = estimator.feature_importances_
    for feature, score in zip(x.columns, importances):
        print(f"{feature:<25} {score:.4f}")
        features.append(feature)

features = list(x)
plt.figure(figsize=(8, 6))
plt.barh(features, importances, color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.yticks(rotation=35, size = 7)
plt.gca().invert_yaxis()
plt.show()


#Plotting and Calculating
#Based on data for comparison of ML's revelations
def averages_calculate(df, group_col, label_col, title_prefix=""):
    grouped = df.groupby(group_col)[label_col].value_counts(normalize=True).unstack().fillna(0)
    grouped.columns = ['Below Average', 'Above Average']
    print(grouped)

#To show ML revelations for underserved group
def bar_graph(df, group_cols, label_col, top_n, title_prefix):
    worst_groups = []
    proportions = []
    for group in group_cols:
        grouped = df.groupby(group)[label_col].value_counts(normalize=True).unstack().fillna(0)
        grouped.columns = ['Below Average', 'Above Average']
        top_worst = grouped['Below Average'].sort_values(ascending=False).head(top_n)
        for idx, val in top_worst.items():
            worst_groups.append(f"{group}: {idx}")
            proportions.append(val)
    plt.ylabel("Proportion Below Average")
    bars = plt.bar(worst_groups, proportions, color='firebrick')
    plt.title(f"{title_prefix}")
    plt.xticks(rotation=35)
    plt.tight_layout()
    plt.show()

# Reverse maps (reapplied for labels)
education_reverse_map = {0: 'No School', 1: 'Primary', 2: 'Secondary', 3: 'High School'}
income_reverse_map = {0: 'Low', 1: 'Medium', 2: 'High'}
employment_reverse_map = {0: 'Farmer', 1: 'Unemployed', 2: 'Student', 3: 'Other', 4: 'Self-Employed'}
age_reverse_map = {0: 'Youth', 1: 'Senior', 2: 'Early Career', 3: 'Midlife'}
engagement_reverse_map = {0: 'Low', 1: 'Medium', 'High': 2}

# Apply readable labels
data['Education_Label'] = data['Education_Level'].map(education_reverse_map)
data['Income_Label'] = data['Household_Income'].map(income_reverse_map)
data['Employment_Label'] = data['Employment_Status'].map(employment_reverse_map)
data['Age_Label'] = data['Age_Group'].map(age_reverse_map)
data['Engagement_Label'] = data['Engagement_Level']

#Get Machine Learning predictions to use for bar graphs to portray models' revelations
xtest_reset = xtest.reset_index(drop=True)
labels_reset = data[['Education_Label', 'Income_Label', 'Employment_Label', 'Age_Label']].iloc[xtest.index].reset_index(drop=True)

xtest_with_labels = pd.concat([xtest_reset, labels_reset], axis=1)

group_data = xtest_with_labels[['Education_Label', 'Income_Label', 'Employment_Label', 'Age_Label']].reset_index(drop=True)

pred_cols = ['Computer_Skill_Pred', 'Internet_Usage_Pred', 'Mobile_Literacy_Pred']
pred_df = ypred_df.copy()
pred_df.columns = pred_cols

ml_data = group_data.copy()
ml_data[pred_cols] = pred_df[pred_cols]

pred_cols_gain = ['Computer_Gain_Pred', 'Internet_Gain_Pred', 'Mobile_Gain_Pred']
pred_df_gain = ypred_gain_df.copy()
pred_df_gain.columns = pred_cols_gain

ml_data_gain = group_data.copy()
ml_data_gain[pred_cols_gain] = pred_df_gain[pred_cols_gain]

# Calling Computer Skill Inequity
print("")
print("Computer Skill Inequity")
print("")
averages_calculate(data, 'Income_Label', 'Computer_Skill_Label', 'ML Predicted Skills')
averages_calculate(data, 'Education_Label', 'Computer_Skill_Label', 'Computer Skill Levels')
averages_calculate(data, 'Employment_Label', 'Computer_Skill_Label', 'Basic_Computer_Knowledge_Score')
averages_calculate(data, 'Age_Label', 'Computer_Skill_Label', 'Basic_Computer_Knowledge_Score')
print("")

print("")
# Calling Internet Usage Inequity
print("Internet Usage Inequity")
print("")
averages_calculate(data, 'Income_Label', 'Internet_Usage_Label', 'Internet_Usage_Score')
averages_calculate(data, 'Education_Label', 'Internet_Usage_Label', 'Internet_Usage_Score')
averages_calculate(data, 'Employment_Label', 'Internet_Usage_Label', 'Internet_Usage_Score')
averages_calculate(data, 'Age_Label', 'Internet_Usage_Label', 'Internet_Usage_Score')
print("")

print("")
# Calling Mobile Literacy Inequity
print("")
print("Mobile Literacy Inequity")
print("")
averages_calculate(data, 'Income_Label', 'Mobile_Literacy_Label', 'Mobile_Literacy_Score')
averages_calculate(data, 'Education_Label', 'Mobile_Literacy_Label', 'Mobile_Literacy_Score')
averages_calculate(data, 'Employment_Label', 'Mobile_Literacy_Label', 'Mobile_Literacy_Score')
averages_calculate(data, 'Age_Label', 'Mobile_Literacy_Label', 'Mobile_Literacy_Score')
print("")

print("")
# Calling Computer Skill Gain Inequity
print("Computer Skill Gain")
print("")
averages_calculate(data, 'Income_Label', 'Computer_Gain_Label', 'Computer_Gain_Label')
averages_calculate(data, 'Education_Label', 'Computer_Gain_Label', 'Computer_Gain_Label')
averages_calculate(data, 'Employment_Label', 'Computer_Gain_Label', 'Computer_Gain_Label')
averages_calculate(data, 'Age_Label', 'Computer_Gain_Label', 'Computer_Gain_Label')
print("")
#
print("")
# Calling Internet Usage Gain Inequity
print("Internet Usage Gain")
print("")
averages_calculate(data, 'Income_Label', 'Internet_Gain_Label', 'Internet_Gain_Label')
averages_calculate(data, 'Education_Label', 'Internet_Gain_Label', 'Internet_Gain_Label')
averages_calculate(data, 'Employment_Label', 'Internet_Gain_Label', 'Internet_Gain_Label')
averages_calculate(data, 'Age_Label', 'Internet_Gain_Label', 'Internet_Gain_Label')
print("")
#
print("")
# Calling Mobile Literacy Gain Inequity
print("Mobile Literacy Gain")
print("")
averages_calculate(data, 'Income_Label', 'Mobile_Gain_Label', 'Mobile_Gain_Label')
averages_calculate(data, 'Education_Label', 'Mobile_Gain_Label', 'Mobile_Gain_Label')
averages_calculate(data, 'Employment_Label', 'Mobile_Gain_Label', 'Mobile_Gain_Label')
averages_calculate(data, 'Age_Label', 'Mobile_Gain_Label', 'Mobile_Gain_Label')
print("")
# Graphs to see ML predictions
features_1 = ["Education_Label", "Income_Label", "Employment_Label", "Age_Label"]

bar_graph(ml_data, features_1, 'Computer_Skill_Pred', 2, 'ML Predictions of Underserved Groups (Basic Computer Knowledge)')
bar_graph(ml_data, features_1, 'Internet_Usage_Pred', 2, 'ML Predictions of Underserved Groups (Internet Usage)')
bar_graph(ml_data, features_1, 'Mobile_Literacy_Pred', 2, 'ML Predictions of Underserved Groups (Mobile Literacy)')

bar_graph(ml_data_gain, features_1, 'Computer_Gain_Pred', 2, 'ML Predictions of Underserved Groups (Basic Computer Knowledge Gain)')
bar_graph(ml_data_gain, features_1, 'Internet_Gain_Pred', 2, 'ML Predictions of Underserved Groups (Internet Usage Gain)')
bar_graph(ml_data_gain, features_1, 'Mobile_Gain_Pred', 2, 'ML Predictions of Underserved Groups (Mobile Literacy Gain)')

#Graphing average values of each group for each of the label
def adaptability(group):
    app_summary = data.groupby(group)['Adaptability_Score'].mean().round(2).sort_values()

    plt.figure(figsize=(10, 6))
    app_summary.plot(kind='barh', color='skyblue')
    plt.title(f'Average Adaptability Score by {group}')
    plt.xlabel('Average Score')
    plt.ylabel(group)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def application(group):
    app_summary = data.groupby(group)['Skill_Application'].mean().round(2).sort_values()

    plt.figure(figsize=(10, 6))
    app_summary.plot(kind='barh', color='skyblue')
    plt.title(f'Average Skill Application Score by {group}')
    plt.xlabel('Average Score')
    plt.ylabel(group)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def literacy(group):
    app_summary = data.groupby(group)['Overall_Literacy_Score'].mean().round(2).sort_values()

    plt.figure(figsize=(10, 6))
    app_summary.plot(kind='barh', color='skyblue')
    plt.title(f'Average Literacy Score by {group}')
    plt.xlabel('Average Score')
    plt.ylabel(group)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


adaptability("Education_Label")
adaptability("Income_Label")
adaptability("Employment_Label")
adaptability("Age_Label")

application("Education_Label")
application("Income_Label")
application("Employment_Label")
application("Age_Label")

literacy("Education_Label")
literacy("Income_Label")
literacy("Employment_Label")
literacy("Age_Label")
