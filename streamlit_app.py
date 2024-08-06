import streamlit as st
import pandas as pd
import os
import tensorflow_decision_forests as tfdf
import numpy as np
os.environ["TF_USE_LEGACY_KERAS"] = "1"


st.title("Jeston Lewis - Capstone Project")


@st.cache_data
def build_data(path):
    data = pd.read_csv(path)
    data.dropna(subset="position", inplace=True)
    data["position"] = data["position"].astype("int")
    return data


@st.cache_resource
def build_model(training):
    train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(training[predictors], label="position")
    model = tfdf.keras.RandomForestModel(verbose=0)
    model.fit(train_ds)
    return model


predictors = [
    "grid", "position", "pos_delta", "driver_code", "constructor_code", "circuit_code", "grid_rolling",
    "position_rolling", "pos_delta_rolling"
]
data = build_data("./data/final_rolling.csv")
training = data[data["year"] < 2022]
test = data[data["year"] >= 2022]
dutch_gp = build_data("./data/dutch_rolling.csv")

test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test[predictors])
dutch_gp_ds = tfdf.keras.pd_dataframe_to_tf_dataset(dutch_gp[predictors])

model = build_model(training)

predictions = model.predict(test_ds)
predictions_df = pd.DataFrame(predictions)

evaluation = model.make_inspector().evaluation()
eval_perc = evaluation.accuracy * 100

st.header(f"Test accuracy - {eval_perc:.2f}%")

preds = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 32, 33
]
full_table = pd.merge(test, predictions_df, on=test.index)
full_table["max_pred"] = full_table[preds].max(axis=1)
single = full_table[
    ["raceId", "driverRef", "constructorRef", "position", "circuitRef", "max_pred"]
].loc[full_table["raceId"] == 1134]
st.table(single)

dutch_pred = model.predict(dutch_gp_ds)
dutch_pred_df = pd.DataFrame(dutch_pred)

dutch_eval = model.make_inspector().evaluation()
dutch_eval_perc = dutch_eval.accuracy * 100

st.header(f"Single test accuracy - {dutch_eval_perc:.2f}%")
dutch_full = pd.merge(dutch_gp, dutch_pred_df, on=dutch_gp.index)

dutch_full["max_pred"] = dutch_full[preds].max(axis=1)

st.table(dutch_full[["driverRef", "constructorRef", "position", "circuitRef", "max_pred"]])

with st.form("Predict a winner"):
    driver_choice = st.selectbox("Driver", full_table["driverRef"].unique())
    circuit_choice = st.selectbox("Circuit", full_table["circuitRef"].unique())
    position_choice = st.selectbox("Position", full_table["position"].unique())
    starting_choice = st.selectbox("Starting", full_table["grid"].unique())

    submit = st.form_submit_button("Predict")

driver_df = None

if submit:
    driver_code = dutch_gp["driver_code"].loc[dutch_gp["driverRef"] == driver_choice]
    constructor_code = dutch_gp["constructor_code"].loc[dutch_gp["driverRef"] == driver_choice]
    circuit_code = dutch_gp["circuit_code"].loc[dutch_gp["driverRef"] == driver_choice]
    grid_rolling = dutch_gp["grid_rolling"].loc[dutch_gp["driverRef"] == driver_choice]
    position_rolling = dutch_gp["position_rolling"].loc[dutch_gp["driverRef"] == driver_choice]
    pos_delta_rolling = dutch_gp["pos_delta_rolling"].loc[dutch_gp["driverRef"] == driver_choice]

    driver_df = pd.DataFrame(
        {
            "grid": [starting_choice],
            "position": [position_choice],
            "pos_delta": [np.nan],
            "driver_code": [driver_code.values[0]],
            "constructor_code": [constructor_code.values[0]],
            "circuit_code": [circuit_code.values[0]],
            "grid_rolling": [grid_rolling.values[0]],
            "position_rolling": [position_rolling.values[0]],
            "pos_delta_rolling": [pos_delta_rolling.values[0]]
        }
    )

if driver_df is not None:
    driver_ds = tfdf.keras.pd_dataframe_to_tf_dataset(driver_df)
    driver_prediction = model.predict(driver_ds)
    driver_prediction_df = pd.DataFrame(driver_prediction)
    driver_full = pd.merge(driver_df, driver_prediction_df, on=driver_df.index)
    driver_full["max_pred"] = driver_full[preds].max(axis=1)
    driver_full["driverRef"] = driver_choice
    st.table(driver_full[["driverRef", "position", "max_pred"]])
