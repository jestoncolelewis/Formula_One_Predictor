import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow_decision_forests as tfdf
import dtreeviz as dt
os.environ["TF_USE_LEGACY_KERAS"] = "1"
st.set_page_config(
    page_title="Jeston Lewis | Capstone",
    layout="wide",
    menu_items=None,
)


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


@st.cache_resource
def build_viz(_local_model, local_data, _local_inspector):
    local_data.dropna(inplace=True)
    features = [f.name for f in _local_inspector.features()]
    viz_model = dt.model(
        model=_local_model,
        X_train=local_data[features],
        y_train=local_data["position"]-1,
        feature_names=features,
        target_name="position",
        tree_index=0
    )
    local_viz = viz_model.view(depth_range_to_display=(0,5), scale=0.75, orientation="LR")
    viz_svg = local_viz.svg()
    return viz_svg


def make_prediction(local_model, local_data):
    local_data_ds = tfdf.keras.pd_dataframe_to_tf_dataset(local_data[predictors])
    predictions = local_model.predict(local_data_ds)
    predictions_df = pd.DataFrame(predictions)
    local_table = pd.merge(local_data, predictions_df, on=local_data.index)
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
    driver_full["driverRef"] = driver_choice
    driver_full["circuit_choice"] = circuit_choice
    return driver_full[["driverRef", "circuit_choice", "position"]]


def update_names(local_table): # TODO move to data_explorer
    driver_refs = local_table["driverRef"].unique()
    circuit_refs = local_table["circuitRef"].unique()
    driver_real = np.array([])
    for driver in driver_refs:
        if driver.find("_") != -1:
            driver = driver.replace("_", " ")
        driver = driver.title()
        driver_real = np.append(driver_real, driver)

    circuit_real = np.array([])
    for circuit in circuit_refs:
        if circuit.find("_") != -1:
            circuit = circuit.replace("_", " ")
        circuit = circuit.title()
        circuit_real = np.append(circuit_real, circuit)

    local_table.replace(driver_refs, driver_real, inplace=True)
    local_table.replace(circuit_refs, circuit_real, inplace=True)
    return local_table


# Data prep
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

# Build model, inspector, and visualization
model = build_model(training)
inspector = model.make_inspector()
viz = build_viz(model, training, inspector)

# Evaluator
evaluation = inspector.evaluation()
eval_perc = evaluation.accuracy * 100

# Single race table
full_table = make_prediction(model, test)
single = update_names(full_table[
    ["driverRef", "grid", "circuitRef", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
].loc[full_table["raceId"] == 1134])
single.sort_values(by="grid", inplace=True)
single["grid"] = single["grid"].astype(int)

# Log plotting
logs = inspector.training_logs()

fig, axs = plt.subplots(2, 1, layout="constrained")

axs[0].plot([log.num_trees for log in logs], [log.evaluation.accuracy for log in logs])
axs[0].set_xlabel("Number of trees")
axs[0].set_ylabel("Accuracy (out-of-bag)")

axs[1].plot([log.num_trees for log in logs], [log.evaluation.loss for log in logs])
axs[1].set_xlabel("Number of trees")
axs[1].set_ylabel("Logloss (out-of-bag)")


# Streamlit elements
st.title("Jeston Lewis - Capstone Project")
tab1, tab2 = st.tabs(["Analysis", "Predictor"])
tab1.title("Single race prediction")
tab1.write(f"Test accuracy - {eval_perc:.2f}%")
tab1.write("Description of below table")
tab1.dataframe(
    single.style.format({1:"{:.2%}", 2:"{:.2%}", 3:"{:.2%}", 4:"{:.2%}", 5:"{:.2%}", 6:"{:.2%}", 7:"{:.2%}", 8:"{:.2%}",
                         9:"{:.2%}", 10:"{:.2%}", 11:"{:.2%}", 12:"{:.2%}", 13:"{:.2%}", 14:"{:.2%}", 15:"{:.2%}",
                         16:"{:.2%}", 17:"{:.2%}", 18:"{:.2%}", 19:"{:.2%}", 20:"{:.2%}"}).highlight_max(),
    use_container_width=True,
    hide_index=True,
    height=738,
) # Single race prediction TODO highlight max better
tab1.title("Visualizations of the model")
col1, col2 = tab1.columns(2)
col1.header("Log plots")
col1.pyplot(plt) # Plot logs
col2.header("Single tree plot")
col2.image(viz, use_column_width=True) # Plot tree

# Prediction form
tab2.title("Race predictor")
tab2.write("Use instructions")
with tab2.form("Predict a winner"):
    f_driver_choice = st.selectbox("Driver", full_table["driverRef"].unique())
    f_circuit_choice = st.selectbox("Circuit", full_table["circuitRef"].unique())
    f_starting_choice = st.selectbox("Start", full_table["grid"].unique())
    f_position_choice = st.selectbox("Finish", full_table["position"].unique())

    submit = st.form_submit_button("Predict")

if submit:
    # TODO remove dutch_gp and make more flexible
    prediction = make_form_prediction(
        driver_choice=f_driver_choice,
        circuit_choice=f_circuit_choice,
        position_choice=f_position_choice,
        starting_choice=f_starting_choice,
        local_data=dutch_gp,
        local_circuits=circuits
    )
    st.dataframe(prediction, use_container_width=True, hide_index=True)
