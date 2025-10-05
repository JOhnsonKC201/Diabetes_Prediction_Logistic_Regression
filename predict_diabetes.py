# ===========================================================================================
# CSIT 265 AI RESEARCH PROJECT
#Unloc# ===========================================================================================
# CSIT 265 AI RESEARCH PROJECT
#Unlocking the Power of Sparse Electronic Health Records (EHR) Data in Developing Nations 
# by Early Type 2 Diabetes Detection Using a Simple Logistic-Regression Model
# Uses DiaHealth dataset (Bangladesh) for type 2 diabetes prediction
# Name : Johnson KC
# Community College of Baltimore County- Essex
# Dr. James Braman And Prof. Lex Brown
# Date : 2025-05-16
# 
 # ===========================================================================================
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer 
from sklearn.model_selection import train_test_split

import os

# Features for the model.
# Biometric features
biometric = ['age', 
             'pulse_rate',
               'systolic_bp',
                 'diastolic_bp', 
                 'glucose', 
                 'height',
                   'weight',
                     'bmi']

# Health condition
flags = ['gender', 
         'family_diabetes', 
         'hypertensive', 
         'family_hypertension', 
         'cardiovascular_disease'
         , 'stroke']

# Features we’ll use
bio_feats = ['age', 'pulse_rate', 'systolic_bp', 'diastolic_bp', 'glucose', 'height', 'weight', 'bmi']
flags = ['gender', 'family_diabetes', 'hypertensive', 'family_hypertension', 'cardiovascular_disease', 'stroke']
all_feats = bio_feats + flags

# ===========================================================================================
#loading the data    
def load_file(file):
    df = pd.read_csv(file)
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0}).fillna(0)

    yn = {'Yes': 1, 'No': 0}
    for f in flags[1:] + ['diabetic']:
        df[f] = df[f].map(yn).fillna(0)

    imp = SimpleImputer(strategy='median')
    df[bio_feats] = imp.fit_transform(df[bio_feats])

    scale = StandardScaler()
    df[bio_feats] = scale.fit_transform(df[bio_feats])

    return df, imp, scale

# Training  the model
def train_it(df):
    X = df[all_feats]
    y = df['diabetic']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=1000)
    model.fit(X_tr, y_tr)

    print(f"Accuracy on test: {model.score(X_te, y_te) * 100:.2f}%")
    return model

# Check the model's accuracy on the training data
def check_model(df, model):
    preds = model.predict(df[all_feats])
    acc = (preds == df['diabetic']).mean()
    print(f"Overall match: {acc * 100:.2f}%")

 # user input       
def get_patient():
    print("Input patient info (skip with Enter):")
    data = {}
    for col, typ in [('age', int), ('pulse_rate', float), ('systolic_bp', float),
                     ('diastolic_bp', float), ('glucose', float), ('height', float),
                     ('weight', float), ('bmi', float)]:
        val = input(f"{col}: ").strip()
        data[col] = typ(val) if val else np.nan

    data['gender'] = 1 if input("Gender (M/F): ").lower().startswith('m') else 0
    for f in flags[1:]:
        ans = input(f"{f.replace('_', ' ')} (Y/N): ").lower()
        data[f] = 1 if ans.startswith('y') else 0

    return pd.DataFrame([data], columns=all_feats)

# ===========================================================================================
#main function
def main():
    csv = next((p for p in ['Diabetes_Final_Data_V2.csv', '/mnt/data/Diabetes_Final_Data_V2.csv'] if os.path.exists(p)), None)
    if not csv:
        print("Where's the file? Couldn't find it.")
        return

    df, imp, scale = load_file(csv)
    model = train_it(df)
    check_model(df, model)

    print("\nNow checking new patient...")
    new_data = get_patient()
    new_data[bio_feats] = imp.transform(new_data[bio_feats])
    new_data[bio_feats] = scale.transform(new_data[bio_feats])

    prob = model.predict_proba(new_data)[0][1]
    print(f"Chance of being diabetic: {prob * 100:.1f}%")
    print("Diagnosis:", "Diabetic" if prob >= 0.3 else "Non diabetic!")

# ===========================================================================================
if __name__ == '__main__':
    main()king the Power of Sparse Electronic Health Records (EHR) Data in Developing Nations 
# by Early Type 2 Diabetes Detection Using a Simple Logistic-Regression Model
# Uses DiaHealth dataset (Bangladesh) for type 2 diabetes prediction
# Name : Johnson KC
# Community College of Baltimore County- Essex
# Dr. James Braman And Prof. Lex Brown
# Date : 2025-05-16
# 
 # ===========================================================================================
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer 
from sklearn.model_selection import train_test_split

import os

# Features for the model.
# Biometric features
biometric = ['age', 
             'pulse_rate',
               'systolic_bp',
                 'diastolic_bp', 
                 'glucose', 
                 'height',
                   'weight',
                     'bmi']

# Health condition
flags = ['gender', 
         'family_diabetes', 
         'hypertensive', 
         'family_hypertension', 
         'cardiovascular_disease'
         , 'stroke']

# Features we’ll use
bio_feats = ['age', 'pulse_rate', 'systolic_bp', 'diastolic_bp', 'glucose', 'height', 'weight', 'bmi']
flags = ['gender', 'family_diabetes', 'hypertensive', 'family_hypertension', 'cardiovascular_disease', 'stroke']
all_feats = bio_feats + flags

# ===========================================================================================
#loading the data    
def load_file(file):
    df = pd.read_csv(file)
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0}).fillna(0)

    yn = {'Yes': 1, 'No': 0}
    for f in flags[1:] + ['diabetic']:
        df[f] = df[f].map(yn).fillna(0)

    imp = SimpleImputer(strategy='median')
    df[bio_feats] = imp.fit_transform(df[bio_feats])

    scale = StandardScaler()
    df[bio_feats] = scale.fit_transform(df[bio_feats])

    return df, imp, scale

# Training  the model
def train_it(df):
    X = df[all_feats]
    y = df['diabetic']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=1000)
    model.fit(X_tr, y_tr)

    print(f"Accuracy on test: {model.score(X_te, y_te) * 100:.2f}%")
    return model

# Check the model's accuracy on the training data
def check_model(df, model):
    preds = model.predict(df[all_feats])
    acc = (preds == df['diabetic']).mean()
    print(f"Overall match: {acc * 100:.2f}%")

 # user input       
def get_patient():
    print("Input patient info (skip with Enter):")
    data = {}
    for col, typ in [('age', int), ('pulse_rate', float), ('systolic_bp', float),
                     ('diastolic_bp', float), ('glucose', float), ('height', float),
                     ('weight', float), ('bmi', float)]:
        val = input(f"{col}: ").strip()
        data[col] = typ(val) if val else np.nan

    data['gender'] = 1 if input("Gender (M/F): ").lower().startswith('m') else 0
    for f in flags[1:]:
        ans = input(f"{f.replace('_', ' ')} (Y/N): ").lower()
        data[f] = 1 if ans.startswith('y') else 0

    return pd.DataFrame([data], columns=all_feats)

# ===========================================================================================
#main function
def main():
    csv = next((p for p in ['Diabetes_Final_Data_V2.csv', '/mnt/data/Diabetes_Final_Data_V2.csv'] if os.path.exists(p)), None)
    if not csv:
        print("Where's the file? Couldn't find it.")
        return

    df, imp, scale = load_file(csv)
    model = train_it(df)
    check_model(df, model)

    print("\nNow checking new patient...")
    new_data = get_patient()
    new_data[bio_feats] = imp.transform(new_data[bio_feats])
    new_data[bio_feats] = scale.transform(new_data[bio_feats])

    prob = model.predict_proba(new_data)[0][1]
    print(f"Chance of being diabetic: {prob * 100:.1f}%")
    print("Diagnosis:", "Diabetic" if prob >= 0.3 else "Non diabetic!")

# ===========================================================================================
if __name__ == '__main__':
    main()
