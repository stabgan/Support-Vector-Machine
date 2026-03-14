# Support Vector Machine (SVM) Classification

Binary classification on a social-network ads dataset using a linear SVM.
Implementations are provided in both Python and R, each producing decision-boundary
visualizations for training and test sets.

## Methodology

1. Load the **Social_Network_Ads.csv** dataset (features: Age, Estimated Salary; target: Purchased).
2. Split into 75 % training / 25 % test.
3. Apply feature scaling (standardization).
4. Train a linear-kernel SVM classifier.
5. Evaluate with a confusion matrix.
6. Plot decision boundaries for both training and test sets.

## Tech Stack

| Layer | Tool |
|-------|------|
| 🐍 Language | Python 3 |
| 📊 Data | pandas, NumPy |
| 🤖 ML | scikit-learn (SVC, StandardScaler) |
| 📈 Visualization | matplotlib |
| 📉 Language (alt) | R |
| 📦 R Packages | caTools, e1071, ElemStatLearn |

## Dependencies

### Python

```
numpy
pandas
matplotlib
scikit-learn
```

Install with:

```bash
pip install numpy pandas matplotlib scikit-learn
```

### R

```r
install.packages(c("caTools", "e1071", "ElemStatLearn"))
```

> **Note:** `ElemStatLearn` has been archived on CRAN. You can install it from
> a mirror or archive:
> `install.packages("ElemStatLearn", repos = "https://cran.r-project.org/src/contrib/Archive/ElemStatLearn/")`

## How to Run

### Python

```bash
python svm.py
```

Two matplotlib windows will open showing the decision boundary for the training
and test sets.

### R

```bash
Rscript svm.R
```

Or open `svm.R` in RStudio and source it.

## Known Issues

- The R implementation depends on **ElemStatLearn**, which is archived on CRAN
  and may require manual installation.
- Visualization uses a fine mesh grid (`step = 0.01`), which can be slow on
  large feature ranges. Increase the step size if performance is a concern.
- The dataset is small (400 rows) — results are illustrative, not
  production-grade.

## License

See [LICENSE](LICENSE) for details.
