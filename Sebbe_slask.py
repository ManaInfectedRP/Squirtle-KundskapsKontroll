# Default Biboltek
import joblib
import time
from math import pi

# Biblotek Numpy
import numpy as np

# Biblotek Seaborn
import seaborn as sns

# Biblotek Pandas
import pandas as pd

# Biblotek Matoplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm

# Biblotek SKLEARN
from sklearn.datasets import fetch_openml, make_regression, load_diabetes, make_moons, make_classification, load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, KFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet, Ridge, Lasso, LassoCV, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor, VotingRegressor, BaggingRegressor, VotingClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, root_mean_squared_error, r2_score, mean_absolute_error, precision_score, recall_score, classification_report, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.svm import LinearSVR, LinearSVC, SVC

# Biblotek xgBoost
import xgboost as xgb

# Biblotek Scipy
from scipy.stats import randint, binom

# Biblotek PIL
from PIL import Image, ImageOps

# Biblotek Streamlit
import streamlit as st

# Ladda in CSV dataset
df = pd.read_csv("(datasetnamn).csv")

# Kolla vilka data typer
print(df.dtypes)

# Kolla om det finns saknade värden
print(df.isnull().sum())

# Beskrivande Statistik
print(df[['(colum name)', '(colum name)']].describe())
