---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: '0.13'
    jupytext_version: '1.16.4'
kernelspec:
  display_name: comp-prob-solv
  language: python
  name: python3
---

# Lecture 28: `scikit-learn`

## Learning Objectives

By the end of this lecture, you will be able to:

- Use `scikit-learn` to perform supervised learning
- Understand the difference between classification and regression
- Train and evaluate classification models
- Train and evaluate regression models

## `scikit-learn`

`scikit-learn` is a Python package that provides simple and efficient tools for data analysis. It is built on `numpy`, `scipy`, and `matplotlib`. It is open source and commercially usable under the BSD license. It is a great tool for machine learning in Python.

### Installation

To install `scikit-learn`, you can follow the instructions on the [official website](https://scikit-learn.org/stable/install.html). You can install it using `pip`:

```bash
pip install -U scikit-learn
```

## Supervised Learning

In supervised learning, we have a dataset consisting of both input features and output labels. The goal is to learn a mapping from the input to the output. We have two types of supervised learning:

1. Classification: The output is a category.
2. Regression: The output is a continuous value.

### Classification

In classification, we have a dataset consisting of input features and output labels. The goal is to learn a mapping from the input features to the output labels. We can use the `scikit-learn` library to perform classification.

#### Machine Learning by Example: Wine Classification

Let's consider an example of wine classification. We have a dataset of wines with different features such as alcohol content, acidity, etc. We want to classify the wines into different categories based on these features.

##### Step 1: Get the Data

First, we need to load the dataset. We can use the `load_wine` function from `sklearn.datasets` to load the wine dataset.

```{code-cell} ipython3
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine

data = load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df.head()
```

```{admonition} Wine Recognition Dataset
:class: tip
The wine recognition dataset is a classic dataset for classification. It contains 178 samples of wine with 13 features each. The features are the chemical composition of the wines, and the target is the class of the wine (0, 1, or 2). You can find more information about the dataset [here](https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-recognition-dataset).
```

```{admonition} Your Data
:class: tip
If `pandas` can read your data, you can swap out the `load_wine` function with `pd.read_csv` or any other method you prefer to load your data.
```

##### Step 2: Explore and Visualize the Data

Next, we need to explore and visualize the data to understand its structure and characteristics. We can use `pandas` to explore the data and `seaborn` to visualize it.

```{code-cell} ipython3
df.describe()
```

```{code-cell} ipython3
df['target'].value_counts()
```

```{code-cell} ipython3
import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(df, hue='target')

plt.show()
```

##### Step 3: Preprocess the Data

Before training the model, we need to preprocess the data. This involves splitting the data into input features and output labels, normalizing the input features, and splitting the data into training and testing sets.

```{code-cell} ipython3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

```{admonition} Why Split the Data?
:class: tip
Splitting the data into training and testing sets allows us to train the model on one set and evaluate it on another set. This helps us assess the model's performance on unseen data. You can also use [cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html) to evaluate the model's performance more robustly.
```

```{admonition} Why Scale or "Standardize" the Data?
:class: tip
Standardizing the data (*e.g.*, using `StandardScaler`) ensures that each feature has a mean of 0 and a standard deviation of 1. This can help improve the performance of some machine learning algorithms, especially those that are sensitive to the scale of the input features.
```

##### Step 4: Train a Model

Now that we have preprocessed the data, we can train a classification model. We will use the `LogisticRegression` and `RandomForestClassifier` models from `scikit-learn`.

```{admonition} Logistic Regression
:class: tip
Logistic regression is a linear model used for binary classification. It models the probability of the output being in a particular category. You can find more information about logistic regression [here](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression).

![Alt text](https://upload.wikimedia.org/wikipedia/commons/c/cb/Exam_pass_logistic_curve.svg)

Canley, CC BY-SA 4.0 <https://creativecommons.org/licenses/by-sa/4.0>, via Wikimedia Commons
```

```{admonition} Random Forest
:class: tip
Random forests are an ensemble learning method that builds multiple decision trees during training and outputs the mode of the classes (*i.e.*, the most frequent class) as the prediction. You can find more information about random forests [here](https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees).

![Alt text](https://media.datacamp.com/legacy/v1718113325/image_7f309c633f.png)

[DataCamp](https://www.datacamp.com/tutorial/random-forests-classifier-python)
```

```{code-cell} ipython3
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Train the Logistic Regression model
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Train the Random Forest model
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Plot the confusion matrix
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ConfusionMatrixDisplay.from_estimator(lr, X_test, y_test, ax=ax[0])
ax[0].set_title('Logistic Regression')
ConfusionMatrixDisplay.from_estimator(rf, X_test, y_test, ax=ax[1])
ax[1].set_title('Random Forest')

plt.show()
```

```{admonition} Confusion Matrix
:class: tip
A confusion matrix is a table that is often used to describe the performance of a classification model. It shows the number of true positives, true negatives, false positives, and false negatives. You can find more information about confusion matrices [here](https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix).
```

##### Step 5: Evaluate the Model

Finally, we need to evaluate the model's performance. We can use metrics such as accuracy, precision, recall, and F1 score to evaluate the model.

```{code-cell} ipython3
from sklearn.metrics import classification_report

print('Logistic Regression:')
print(classification_report(y_test, y_pred_lr))

print('Random Forest:')
print(classification_report(y_test, y_pred_rf))
```

```{admonition} Classification Report
:class: tip
A classification report shows the precision, recall, F1 score, and support for each class in the classification model. Precision is the ratio of true positives to the sum of true positives and false positives

$$
\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}
$$

Recall is the ratio of true positives to the sum of true positives and false negatives

$$
\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}
$$

The F1 score is the harmonic mean of precision and recall

$$
F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

Support is the number of occurrences of each class in the dataset.
```

##### Step 6: Plot and Interpret the Coefficients

For the logistic regression model, we can plot and interpret the coefficients to understand the importance of each feature in the classification.

```{code-cell} ipython3
import numpy as np

# Ensure feature names are a NumPy array
feature_names = np.array(data.feature_names)

# Sort the coefficients
sorted_idx = lr.coef_[0].argsort()

# Plot the coefficients
plt.figure(figsize=(12, 6))
plt.barh(feature_names[sorted_idx], lr.coef_[0][sorted_idx])
plt.xlabel('Coefficient Value')
plt.ylabel('Feature Name')
plt.title('Logistic Regression Coefficients')
plt.show()
```

The plot above shows the coefficients of the logistic regression model. The features with the largest coefficients (in absolute value) are the most important for the classification. The sign of the coefficient indicates the direction of the relationship between the feature and the target. The two features with the largest coefficients are `proline` and `alcalinity_of_ash`.

`proline` is the amount of proline in the wine. Proline is an amino acid that is found in high concentrations in red wines. The coefficient for `proline` is positive, indicating that wines with higher proline content are more likely to be classified as class 2.

```{figure} https://upload.wikimedia.org/wikipedia/commons/4/45/L-Proline.svg
---
width: 200px
name: proline
---
The chemical structure of proline. By Qohelet12, CC0, via Wikimedia Commons
```

`alcalinity_of_ash` is the amount of ash in the wine. Ash is the inorganic residue remaining after the water and organic matter have been removed by heating. The coefficient for `alcalinity_of_ash` is negative, indicating that wines with lower ash content are more likely to be classified as class 2.

```{admonition} Your Data
:class: tip
You can swap out the wine dataset with your own dataset to perform classification on your data. Make sure to preprocess the data, train the model, and evaluate the model as shown in the example above.
```

### Regression

Under construction...
