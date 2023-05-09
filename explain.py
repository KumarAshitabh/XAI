#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from eli5.sklearn import PermutationImportance
import eli5
import shap
import pickle
import lime
import lime.lime_tabular
import os
import streamlit as st
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource

def perm_import(
    model,
    X_val,
    y_val,
    score,
    return_importances=False,
    ):

    # Load up model

    ml_model = pickle.load(open(model, 'rb'))
    perm = PermutationImportance(ml_model, scoring=score,
                                 random_state=1).fit(X_val, y_val)
    feat_name = X_val.columns.tolist()
    eli5_show_weights = eli5.show_weights(perm, feature_names=feat_name)

    importances = eli5.explain_weights_df(perm, feature_names=feat_name)

    if return_importances == True:
        return importances


def shapValue(
    model,
    x_train,
    x_val,
    tree_model,
    row_to_show=5,
    ):

    # open ml_model

    ml_model = pickle.load(open(model, 'rb'))
    X_train_summary = shap.kmeans(x_train, 50)
    data_for_prediction = x_val.iloc[row_to_show]
    data_for_prediction_array = data_for_prediction.values.reshape(1,
            -1)

    # when using tree model

    if tree_model:
        try:
            explainer = shap.TreeExplainer(ml_model)
            shap_values = explainer.shap_values(data_for_prediction)
            shap.initjs()
            return shap.force_plot(explainer.expected_value[1],
                                   shap_values[1], data_for_prediction,
                                   matplotlib=True, show=False)
        except Exception as e:
            print (e)
    else:

        explainer = shap.KernelExplainer(ml_model.predict_proba,
                X_train_summary)
        shap_values = explainer.shap_values(data_for_prediction)
        return shap.force_plot(explainer.expected_value[1],
                               shap_values[1], data_for_prediction,
                               matplotlib=True, show=False)



def lime_explain(x_train, x_val, y_train, feat, model, i):
    ml_model = pickle.load(open(model, 'rb'))
    explainer = lime.lime_tabular.LimeTabularExplainer(x_train.values, 
                                                        feature_names = feat, 
                                                        class_names = y_train.iloc[:,0].unique(), 
                                                        mode='classification', 
                                                        training_labels=x_train.columns.values.tolist())
            
    predict_fn = lambda x: ml_model.predict_proba(x).astype(float)
    exp = explainer.explain_instance(x_val.values[i], predict_fn, num_features = 5)
    exp.save_to_file('lime.html')
