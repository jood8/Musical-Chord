
import librosa
import numpy as np
import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report,ConfusionMatrixDisplay,confusion_matrix
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score,GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import joblib

#extracte_data
def extracte_data(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050, duration=4.0) 
        if len(y) < 2048: 
                return None
        chroma = librosa.feature.chroma_cens(y=y, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr,n_fft=512)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        combined = np.vstack((chroma, contrast, mfcc)) 
        
        return np.concatenate([
            np.mean(combined.T, axis=0), 
            np.std(combined.T, axis=0)
        ]) 
    except Exception as e:
        return None
    
base_path = r"Audio_Files" 
files = glob.glob(os.path.join(base_path, "**/*.wav"), recursive=True)

data,labels = [] , []

for file in files:
    feature = extracte_data(file)
    if feature is not None:
        data.append(feature)
        
        label = os.path.basename(os.path.dirname(file))
        labels.append(label)
        
X = np.array(data)
y=np.array(labels)
print(X.shape)

#encoding
le = LabelEncoder()
y_encoded = le.fit_transform(labels)

#train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, train_size=0.8, random_state=42, stratify=y_encoded)

#StandardScaler
scaler = StandardScaler()
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y_encoded)

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, train_size=0.8, random_state=42, stratify=y_resampled
)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#MODLING
model_params = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1000, random_state=42),
        "params": {"C": [0.1, 1, 10, 100], "solver": ["lbfgs"]}
    },
    "KNN": {
        "model": KNeighborsClassifier(),
        "params": {"n_neighbors": [5]}},
    
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {"n_estimators": [100], "max_depth": [None]}
    },
    
    "SVM": {
    "model": SVC(probability=True, random_state=42),
    "params": {
        "C": [0.1, 1, 10, 100],
        "kernel": ["rbf", "poly"],  
        "gamma": ["scale", "auto"]
    }
},
    "XGBoost": {
        "model": XGBClassifier(eval_metric='mlogloss', random_state=42),
        "params": {"n_estimators": [50], "learning_rate": [0.1]}
    }, 
}


print("\n--- Comparison and optimization phase ---")
best_models = {}
results = {}

for name, mp in model_params.items():
    clf = GridSearchCV(mp["model"], mp["params"], cv=10, n_jobs=-1, verbose=0)
    clf.fit(X_train_scaled , y_train)
    best_models[name] = clf.best_estimator_
    results[name] = clf.best_score_
    print(f"✅ {name}: the accuracy% = {clf.best_score_*100:.2f}%")

winner_name = max(results, key=results.get)
final_model = best_models[winner_name]

print(f"The Best Model: {winner_name} accuracy% {results[winner_name]*100:.2f}")
print("-"*7)

final_preds = final_model.predict(X_test_scaled)
test_acc = accuracy_score(y_test, final_preds)
print(f"Accuracy_TestData: {test_acc*100:.4f}%")

#SAVE MODEL
final_pipeline = Pipeline([
    ("scaler", scaler),
    ("model", final_model)
])

joblib.dump({"pipeline": final_pipeline, "label_encoder": le}, "chord_pipeline.pkl")

print(f"The winner was {winner_name} with test accuracy: {test_acc*100:.2f}%")


y_pred = final_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap='Blues', xticks_rotation='vertical')
plt.title(f'Confusion Matrix for {winner_name}')
plt.show()
print("\n--- Detailed Classification Report ---")
print(classification_report(y_test, y_pred, target_names=le.classes_))