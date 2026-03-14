# Support Vector Machine — Social Network Ads

Binary classification using a linear SVM to predict whether a user purchases a product based on age and estimated salary.

## What It Does

Trains a Support Vector Machine with a linear kernel on the [Social Network Ads](Social_Network_Ads.csv) dataset, evaluates performance with a confusion matrix and classification report, and visualises the decision boundary for both training and test sets.

## Dataset

| Column           | Description                        |
|------------------|------------------------------------|
| User ID          | Unique identifier (not used)       |
| Gender           | Male / Female (not used)           |
| Age              | User age                           |
| EstimatedSalary  | Annual estimated salary            |
| Purchased        | 0 = No, 1 = Yes (target)          |

400 samples, 75/25 train-test split.

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| 🐍 Python 3.8+ | Primary language |
| 📊 scikit-learn | SVM classifier, preprocessing, metrics |
| 🔢 NumPy | Numerical operations |
| 🐼 pandas | Data loading |
| 📈 matplotlib | Decision boundary visualisation |
| 📊 R | Alternative implementation (`svm.R`) |

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Run the classifier
python svm.py
```

The R version requires the `caTools` and `e1071` packages:

```r
install.packages(c("caTools", "e1071"))
source("svm.R")
```

## Project Structure

```
├── svm.py                  # Python implementation
├── svm.R                   # R implementation
├── Social_Network_Ads.csv  # Dataset
├── requirements.txt        # Python dependencies
└── README.md
```

## ⚠️ Known Issues

- The R script depends on `ElemStatLearn`, which has been archived on CRAN. Use `install.packages("ElemStatLearn", repos = "https://cran.r-project.org/src/contrib/Archive/ElemStatLearn/")` or replace the visualisation with `ggplot2`.
- Only a linear kernel is used; an RBF kernel may yield better accuracy on this dataset.
