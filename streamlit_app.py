import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow_decision_forests as tfdf
os.environ["TF_USE_LEGACY_KERAS"] = "1"


@st.cache_data
def build_data(path):
    local_data = pd.read_csv(path)
    local_data.dropna(subset="position", inplace=True)
    local_data["position"] = local_data["position"].astype("int")
    return local_data


@st.cache_resource
def build_model(local_training):
    train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(local_training[predictors], label="position")
    local_model = tfdf.keras.RandomForestModel(verbose=0)
    local_model.fit(train_ds)
    return local_model


def make_prediction(local_model, local_data):
    local_data_ds = tfdf.keras.pd_dataframe_to_tf_dataset(local_data[predictors])
    predictions = local_model.predict(local_data_ds)
    predictions_df = pd.DataFrame(predictions)
    local_table = pd.merge(local_data, predictions_df, on=local_data.index)
    local_table["max_pred"] = local_table[preds].max(axis=1)
    driverRefs = local_table["driverRef"].unique()
    circuitRefs = local_table["circuitRef"].unique()
    constructorRefs = local_table["constructorRef"].unique()
    driverReal = np.array([
        "Charles Leclerc", "Carlos Sainz", "Lewis Hamilton", "George Russell","Kevin Magnussen", "Valtteri Bottas",
        "Esteban Ocon", "Yuki Tsunoda", "Fernando Alonso", "Zhou Guanyu", "Mick Schumacher", "Lance Stroll",
        "Alex Albon", "Daniel Ricciardo", "Lando Norris", "Nicholas Latifi", "Nico Hulkenberg", "Checo Perez",
        "Max Verstappen", "Pierre Gasly", "Sebastian Vettel", "Nyk De Vries", "Logan Sargeant", "Oscar Piastri",
        "Liam Lawson", "Oliver Bearman"
    ])
    circuitReal = np.array([
        "Bahrain", "Jeddah", "Albert Park", "Imola", "Miami", "Catalunya", "Monaco", "Baku", "Villeneuve",
        "Silverstone", "Red Bull Ring", "Ricard", "Hungaroring", "Spa", "Zandvoort", "Monza", "Marina Bay", "Suzuka",
        "Circuit of the Americas", "Rodriquez", "Interlagos", "Yas Marina", "Losail", "Las Vegas", "Shanghai"
    ])
    constructorReal = np.array([
        "Ferrari", "Mercedes", "Haas", "Alfa Romeo", "Alpine", "Alphatauri", "Aston Martin", "Williams", "Mclaren",
        "Red Bull", "Sauber", "RB"
    ])
    local_table.replace(driverRefs, driverReal, inplace=True)
    local_table.replace(circuitRefs, circuitReal, inplace=True)
    local_table.replace(constructorRefs, constructorReal, inplace=True)
    return local_table


def make_form_prediction(driver_choice, circuit_choice, position_choice, starting_choice, local_data, local_circuits):
    driver_code = local_data["driver_code"].loc[local_data["driverRef"] == driver_choice].values[0]
    constructor_code = local_data["constructor_code"].loc[local_data["driverRef"] == driver_choice].values[0]
    circuit_code = local_circuits["circuit_code"].loc[local_circuits["circuitRef"] == circuit_choice].values[0]
    grid_rolling = local_data["grid_rolling"].loc[local_data["driverRef"] == driver_choice].values[0]
    position_rolling = local_data["position_rolling"].loc[local_data["driverRef"] == driver_choice].values[0]
    pos_delta_rolling = local_data["pos_delta_rolling"].loc[local_data["driverRef"] == driver_choice].values[0]
    pos_delta = starting_choice - position_choice
    driver_df = pd.DataFrame(
        {
            "driver_code": [driver_code],
            "constructor_code": [constructor_code],
            "circuit_code": [circuit_code],
            "grid_rolling": [grid_rolling],
            "position_rolling": [position_rolling],
            "pos_delta_rolling": [pos_delta_rolling],
            "grid": [starting_choice],
            "position": [position_choice],
            "pos_delta": [pos_delta]
        }
    )
    driver_ds = tfdf.keras.pd_dataframe_to_tf_dataset(driver_df)
    driver_prediction = model.predict(driver_ds)
    driver_prediction_df = pd.DataFrame(driver_prediction)
    driver_full = pd.merge(driver_df, driver_prediction_df, on=driver_df.index)
    driver_full["max_pred"] = driver_full[preds].max(axis=1)
    driver_full["driverRef"] = driver_choice
    driver_full["circuit_choice"] = circuit_choice
    st.table(driver_full[["driverRef", "circuit_choice", "position", "max_pred"]])


data = build_data("./data/final_rolling.csv")
training = data[data["year"] < 2022]
test = data[data["year"] >= 2022]
dutch_gp = build_data("./data/dutch_rolling.csv")
circuits = pd.DataFrame()
circuits["circuit_code"] = data["circuit_code"].unique()
circuits["circuitRef"] = data["circuitRef"].unique()
predictors = [
    "grid", "position", "pos_delta", "driver_code", "constructor_code", "circuit_code", "grid_rolling",
    "position_rolling", "pos_delta_rolling"
]
preds = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
]

st.title("Jeston Lewis - Capstone Project")

# Build model and inspector
model = build_model(training)
inspector = model.make_inspector()

# Evaluator
evaluation = inspector.evaluation()
eval_perc = evaluation.accuracy * 100
st.header(f"Test accuracy - {eval_perc:.2f}%")

# Single race table
full_table = make_prediction(model, test)
single = full_table[
    ["driverRef", "constructorRef", "grid", "circuitRef", 1]
].loc[full_table["raceId"] == 1134]
single.sort_values(by="grid", inplace=True)
single["grid"] = single["grid"].astype(int)
st.dataframe(single.style.format({1:"{:.2%}"}), use_container_width=True, hide_index=True)

# Log plotting
# TODO: come up with unique things to plot
logs = inspector.training_logs()

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot([log.num_trees for log in logs], [log.evaluation.accuracy for log in logs])
plt.xlabel("Number of trees")
plt.ylabel("Accuracy (out-of-bag)")

plt.subplot(1, 2, 2)
plt.plot([log.num_trees for log in logs], [log.evaluation.loss for log in logs])
plt.xlabel("Number of trees")
plt.ylabel("Logloss (out-of-bag)")

st.pyplot(plt)

# Plot tree
tree = inspector.extract_tree(tree_idx=0)
st.write(tree)

# Prediction form
with st.form("Predict a winner"):
    f_driver_choice = st.selectbox("Driver", full_table["driverRef"].unique())
    f_circuit_choice = st.selectbox("Circuit", full_table["circuitRef"].unique())
    f_position_choice = st.selectbox("Position", full_table["position"].unique())
    f_starting_choice = st.selectbox("Starting", full_table["grid"].unique())

    submit = st.form_submit_button("Predict")

    if submit:
        make_form_prediction(f_driver_choice, f_circuit_choice, f_position_choice, f_starting_choice, dutch_gp, circuits)
