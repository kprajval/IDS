import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib

def preprocess_data(filepath):
    df = pd.read_csv(filepath)
    X = df.drop(columns=["Label", "Dst IP", "Src IP"])
    X = X.dropna()
    y = df["Label"].loc[X.index]

    # Visualize original distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(y)
    plt.title("Original Class Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("original_class_distribution.png")
    plt.show()

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    return X_train_smote, X_test, y_train_smote, y_test, le

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, label_encoder, model_name):
    y_pred = model.predict(X_test)
    print(f"Classification Report for {model_name}:")
    print(classification_report(y_test, y_pred))

    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{model_name.lower()}_confusion_matrix.png")
    plt.show()

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for {model_name}: {accuracy:.4f}")
    return accuracy

def train_ensemble_model(models, X_train, y_train):
    voting_clf = VotingClassifier(estimators=models, voting='hard')
    voting_clf.fit(X_train, y_train)
    return voting_clf

# ----------- Main Script -----------

X_train, X_test, y_train, y_test, label_encoder = preprocess_data("OD-IDS2022-Dataset.csv")

rf_model = train_random_forest(X_train, y_train)
xgb_model = train_xgboost(X_train, y_train)
dt_model = train_decision_tree(X_train, y_train)

# Save individual models
joblib.dump(rf_model, "random_forest_model.joblib")
joblib.dump(xgb_model, "xgboost_model.joblib")
joblib.dump(dt_model, "decision_tree_model.joblib")

evaluate_model(rf_model, X_test, y_test, label_encoder, "Random Forest")
evaluate_model(xgb_model, X_test, y_test, label_encoder, "XGBoost")
evaluate_model(dt_model, X_test, y_test, label_encoder, "Decision Tree")

# Ensemble
ensemble = train_ensemble_model(
    models=[("rf", rf_model), ("xgb", xgb_model), ("dt", dt_model)],
    X_train=X_train,
    y_train=y_train
)

evaluate_model(ensemble, X_test, y_test, label_encoder, "Ensemble Model")
joblib.dump(ensemble, "ensemble_model.joblib")
print("Ensemble model saved as ensemble_model.joblib")
