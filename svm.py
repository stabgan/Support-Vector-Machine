# Support Vector Machine (SVM)
# Classification on Social Network Ads dataset

import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report


def load_data():
    """Load the Social Network Ads dataset relative to script location."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "Social_Network_Ads.csv")
    dataset = pd.read_csv(csv_path)
    X = dataset.iloc[:, [2, 3]].values  # Age, EstimatedSalary
    y = dataset.iloc[:, 4].values       # Purchased
    return X, y


def train_svm(X_train, y_train):
    """Fit a linear SVM classifier."""
    classifier = SVC(kernel="linear", random_state=0)
    classifier.fit(X_train, y_train)
    return classifier


def evaluate(classifier, X_test, y_test):
    """Print confusion matrix and classification report."""
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Not Purchased", "Purchased"]))
    return y_pred


def plot_decision_boundary(classifier, X_set, y_set, title):
    """Visualise the SVM decision boundary."""
    colors = ("red", "green")
    cmap_bg = ListedColormap(colors)

    X1, X2 = np.meshgrid(
        np.arange(X_set[:, 0].min() - 1, X_set[:, 0].max() + 1, 0.01),
        np.arange(X_set[:, 1].min() - 1, X_set[:, 1].max() + 1, 0.01),
    )
    plt.figure(figsize=(8, 6))
    plt.contourf(
        X1, X2,
        classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
        alpha=0.75,
        cmap=cmap_bg,
    )
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())

    for idx, label in enumerate(np.unique(y_set)):
        plt.scatter(
            X_set[y_set == label, 0],
            X_set[y_set == label, 1],
            c=colors[idx],
            label=label,
            edgecolors="k",
            alpha=0.7,
        )

    plt.title(title)
    plt.xlabel("Age")
    plt.ylabel("Estimated Salary")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    # Load data
    X, y = load_data()

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    # Feature scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Train SVM
    classifier = train_svm(X_train, y_train)

    # Evaluate
    evaluate(classifier, X_test, y_test)

    # Visualise
    plot_decision_boundary(classifier, X_train, y_train, "SVM (Training set)")
    plot_decision_boundary(classifier, X_test, y_test, "SVM (Test set)")


if __name__ == "__main__":
    main()
