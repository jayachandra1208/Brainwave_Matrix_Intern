# Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Load Data
df = pd.read_csv('C:/Users/koona/OneDrive/Pictures/JAYACHANDRA/Brainwave_Matrix_Intern/TASK 2/creditcardfraud/creditcard.csv')

# Inspect the Dataset
print(df.info())
print(df['Class'].value_counts())

# Separate Features and Target
X = df.drop(columns=['Class'])
y = df['Class']

# Scale 'Amount' and 'Time'
scaler = StandardScaler()
X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle Imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Train Model
model = RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced')
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate Model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
