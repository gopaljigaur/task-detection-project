import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import joblib
import os
import pickle as pkl


def load_data(base_path:str):
    class_mapping = {}
    running_count = 0
    filter_fn = lambda name: ".obj_agg" in name
    data = []
    labels = []
    for task in os.listdir(base_path):
        class_mapping[task] = running_count
        task_path = os.path.join(base_path, task)
        files = os.listdir(task_path)
        filtered_files = list(filter(filter_fn, files))
        for file in filtered_files:
            tensor = torch.load(os.path.join(task_path,file))
            data.append(tensor.tolist())
            labels.append(running_count)
        running_count += 1

    return [pd.DataFrame(data), pd.DataFrame(labels), class_mapping]



if __name__ == "__main__":
    [X,y, class_mapping] = load_data("training_data/training_multiple")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Create the Decision Tree model
    dt_model = DecisionTreeClassifier(random_state=42)

    # Train the model using the training data
    dt_model.fit(X_train, y_train)
    # Make predictions on the testing set
    y_pred = dt_model.predict(X_test)

    print(class_mapping)
    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Additional metrics and details
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Provide a file path where you want to save the model
    file_path = "training_data/decision_tree_model.pkl"

    # Save the model using joblib
    joblib.dump(dt_model, file_path)