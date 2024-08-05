import streamlit as st
import pandas as pd
import ydf

st.write("Jeston Lewis - Capstone Project")

data = pd.read_csv("./data/final_rolling.csv")
predictors = ["grid", "position", "pos_delta", "driver_code", "constructor_code", "circuit_code", "grid_rolling", "position_rolling", "pos_delta_rolling"]
data.dropna(subset="position", inplace=True)
data["position"] = data["position"].astype("int")

training = data[data["year"] < 2022]
test = data[data["year"] >= 2022]

model = ydf.RandomForestLearner(label="position").train(training[predictors])

predictions = model.predict(test[predictors])
predictions_df = pd.DataFrame(predictions)

evaluation = model.evaluate(test[predictors])

st.write(f"test accuracy {evaluation.accuracy}")

full_table = pd.merge(test, predictions_df, on=test.index)
full_table["max_pred"] = full_table[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]].max(axis=1)

st.table(full_table[["raceId", "driverRef", "constructorRef", "position", "circuitRef", "max_pred"]])
