import streamlit as st
import pandas as pd
import ydf

st.write("Lewis Jeston - Capstone Project")

dataset_df = pd.read_csv("./data/final_rolling.csv")

training = dataset_df[dataset_df["raceId"] < 2022]
test = dataset_df[dataset_df["year"] >= 2022]

model = ydf.RandomForestLearner(label="position").train(training)

st.write(model.predict(test))
