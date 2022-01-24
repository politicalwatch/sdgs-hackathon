# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# %%
df = pd.read_csv("final_features.csv", dtype = np.uint8)

# %%
df.head()

# %%
targets = df.filter(like='ODS')
features = df[list(set(df.columns).difference(targets))]

# %%
features.head()

# %%
targets.head()

# %%
del df

# %% [markdown]
# # Modelling

# %%
models = [RandomForestClassifier() for c in targets.columns]
#models = [GaussianNB() for c in targets.columns]

# %%
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.5, random_state=0)
models = {}
i = 0
for category in targets.columns:
    print(i)
    models[category] = GaussianNB().fit(X_train, y_train[category])
    i+=1

# %%
results = {}
for category in targets.columns:
    preds = models[category].predict(X_test)
    print(roc_auc_score(y_test, preds.reshape(-1, 1)))
    results[category] = roc_auc_score(y_test, preds.reshape(-1, 1))

# %%
np.mean(list(results.values()))

# %%
results

# %%
pd.DataFrame(targets.sum())
