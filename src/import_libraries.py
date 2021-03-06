def import_libraries():
    import os

    import numpy as np

    import collections

    import time

    from random import randint

    from matplotlib import pyplot as plt
    from matplotlib.offsetbox import AnchoredText
    from matplotlib.pyplot import figure
    from matplotlib import cm
    from colorspacious import cspace_converter
    #%matplotlib inline

    import seaborn as sns
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 2,'font.family': [u'times']})

    import pandas as pd

    import scipy.stats

    import warnings
    warnings.filterwarnings("ignore")

    from sklearn import preprocessing
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_regression
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import r2_score
    from sklearn import linear_model
    from sklearn.linear_model import LinearRegression, Lasso, Ridge
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_regression, f_classif
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, accuracy_score
    from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score, roc_curve, auc


    # Models used in classification:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
    from sklearn.svm import SVC
    from xgboost import XGBClassifier
    from sklearn.tree import DecisionTreeClassifier
    from catboost import CatBoostClassifier

    # Over/under sampling techniques:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler

    import lightgbm as lgb

    from lazypredict.Supervised import LazyClassifier

    from tabulate import tabulate

    # PyTorch
    import torch
    import torch.nn as nn
    import torchvision
    import matplotlib.pyplot as plt
    import torch.nn.functional as F
    import tqdm
