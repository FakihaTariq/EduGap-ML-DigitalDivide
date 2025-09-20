# EduGap-AI-DigitalDivide
An AI powered tool to analyze digital readiness and evaluate efficiency and results of interventions for bridging the digital divide, using machine learning, demographic/socioeconomic, behavioral, and pre and post training skills data.

# Abstract
The digital divide remains a persistent and evolving barrier to equity in the modern age, particularly in rural and semi-rural populations. While most research highlights disparities in broadband access and device availability, especially for income and geographic factors, fewer studies explore skill-based inequity in digital readiness across socioeconomic and demographic groups such as education, employment, age, and income. This study introduces a machine learning driven framework designed to predict and analyze digital skill readiness and skill gains after interventions across three core areas: basic computer knowledge, internet usage, and mobile literacy. Using a synthetic dataset representative of populations in underserved regions, multiple models were evaluated to classify individuals as above or below average in each skill and skill gain, though for simplicity purposes the terms above and below average are used even though the median is used as the splitting point, not the mean. The median is used in order to improve equity and balance between the classes. A final pipeline using a MultiOutputClassifier with a RandomForest base achieved a testing accuracy of around .6, much greater from initial accuracies near .35, while avoiding overfitting. The model was further used to identify population subgroups most at risk and assess their growth potential after digital literacy training. Results show that groups with lower formal education, youth, early career, unemployed, informal work, high and low income were among the most digitally underserved. However, several of these same groups showed signs of above average gain, indicating strong potential for intervention impact though other groups continued to have below average gain showing that those groups may need extra support. Average adaptability, skill application, and overall literacy scores indicate that future interventions need to integrate techniques to target the areas they are struggling in. This work demonstrates how machine learning can complement traditional digital divide research and be a foundation for more scalable, ongoing, and valuable assets for future digital readiness and program effectiveness research.

# Research Paper
Access the pdf through 'EduGap_Research_Paper.pdf'

# Project Objective/Goals
- This project aims to predict readiness using demographic, socioeconmic, skill, and behvaioral features
- Label underserved comuntiites based on equity focused metrics
- Analyze and visualize digital inequity through analysis and graphs
- Assess the effectiveness of digital inclusion interventions

# Repository Contents
- 'EduGap_Analysis_Notebook.ipynb': Jupyter notebook with code snippets, visualizations, and model training
- 'digital_literacy_dataset.csv': Dataset used for training and evaluation of the models
- 'EduGap_Research_Paper.pdf': Final research paper describing methodology and findings
- 'ML.py': Full code script for the project

# Features
  - Uses both median based and quantile based labeling for digital equity analysis
  - Incorporates custom metrics for access and skills categorization
  - Random Forest classification model with optimized hyperparameters
  - Multi-output classification to predict multiple readiness indicators
  - Visualizations and analysis calculations that highlight gaps in access, skills, and outcomes
 
# Dataset Description
This dataset includes:
  - Demographic/Socioeconomic Factors: Age, gender, education level, employment status, etc
  - Skill Scores: Computer, mobile, and internet skill scores pre and post training
  - Engagement/Behvaioral Metrics: Engagement level, session count, quiz performance, etc

# Model Summary
- Model: Multi Output Classifier with Random Forest Classifier base model
- Labels: Binary labels (0: Below Average Skills/Skill Gain, 1: Above Average Skills/Skill Gain)
- Input Features: Normalized categorical features 
- Evaluation: Accuracy, classification report

# Visualizations/Analysis
- Feature importance
- Identifying most underserved groups
- Calculated amount below "average" (median) and amount above

# How to Run
1. Clone the repository:
git clone https://github.com/FakihaTariq/EduGap-AI-DigitalDivide.git
cd edugap
2. Install Required Packages:
pip install -r requirements.txt
If pip doesn't work, try:
python -m pip install -r requirements.txt
3. Launch Jupyter Notebook
If using IDE with Jupyter support: jupyter notebook
And open EduGap_Analysis_Notebook.ipynb in your browser
or python script:
python ML.py
4. Add Dataset 
Place your csv file in appropriate directory and access it with correct name
5. Run cells/script
