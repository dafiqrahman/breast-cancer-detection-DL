import pickle
import numpy as np
import pandas as pd


def predict(model, data):
    y_pred = model.predict_proba(data)
    y_pred = np.round(y_pred, 3)
    return y_pred
