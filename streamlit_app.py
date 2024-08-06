import streamlit as st
import pandas as pd
import ydf
import numpy as np


st.title("Jeston Lewis - Capstone Project")

@st.cache_data
def build_data():
    data = pd.read_csv("./data/final_rolling.csv")
    data.dropna(subset="position", inplace=True)
    data["position"] = data["position"].astype("int")
    return data

@st.cache_resource
def build_model():
    return ydf.RandomForestLearner(label="position").train(training[predictors])

predictors = ["grid", "position", "pos_delta", "driver_code", "constructor_code", "circuit_code", "grid_rolling", "position_rolling", "pos_delta_rolling"]
data = build_data()
training = data[data["year"] < 2022]
test = data[data["year"] >= 2022]
dutch_gp = pd.DataFrame(
    {
        "grid": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        "position": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "pos_delta": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        "driver_code": [343, 617, 461, 519, 580, 699, 607,  20, 583, 670, 756,  14, 290,
       425, 102, 798, 705, 379, 855, 693],
        "constructor_code": [136, 131,  69, 165, 131,  69, 165,  11,   5, 162,  11, 208,   5,
        79, 168, 162, 208,  79, 168, 136],
        "circuit_code": [39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39],
         "grid_rolling": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
         "position_rolling": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
         "pos_delta_rolling": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    }
)

model = build_model()

predictions = model.predict(test[predictors])
predictions_df = pd.DataFrame(predictions)

evaluation = model.evaluate(test[predictors])

st.header(f"Test accuracy - {evaluation.accuracy:.2f}%")

full_table = pd.merge(test, predictions_df, on=test.index)
full_table["max_pred"] = full_table[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]].max(axis=1)

single = full_table[["raceId", "driverRef", "constructorRef", "position", "circuitRef", "max_pred"]].loc[full_table["raceId"] == 1134]
st.table(single)

dutch_pred = model.predict(dutch_gp)
dutch_pred_df = pd.DataFrame(dutch_pred)

dutch_eval = model.evaluate(dutch_gp)

st.header(f"Single test accuracy - {dutch_eval.accuracy:.2f}%")
dutch_full = pd.merge(dutch_gp, dutch_pred_df, on=dutch_gp.index)

dutch_full["max_pred"] = dutch_full[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]].max(axis=1)
st.table(dutch_full)