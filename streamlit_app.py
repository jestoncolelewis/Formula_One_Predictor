import streamlit as st
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_decision_forests as tfdf
import dtreeviz as dt

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
def load_saved_model(path):
    try:
        return tf.keras.models.load_model(path)
    except:
        return None


@st.cache_resource
def build_model(local_training):
    train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(local_training[predictors], label="position")
    local_model = tfdf.keras.RandomForestModel()
    local_model.fit(train_ds)
    local_model.save("./model")
    return local_model


def build_viz(local_model, local_data, local_inspector):
    local_data.dropna(inplace=True)
    features = [f.name for f in local_inspector.features()]
    viz_model = dt.model(
        model=local_model,
        X_train=local_data[features],
        y_train=local_data["position"]-1,
        feature_names=features,
        target_name="position",
        tree_index=0
    )
    local_viz = viz_model.view(depth_range_to_display=(0,3), scale=0.75, orientation="LR")
    viz_svg = local_viz.svg()
    return viz_svg


def make_prediction(local_model, local_data):
    local_data_ds = tfdf.keras.pd_dataframe_to_tf_dataset(local_data[predictors])
    predictions = local_model.predict(local_data_ds)
    predictions_df = pd.DataFrame(predictions)
    local_table = pd.merge(local_data, predictions_df, on=local_data.index)
    return local_table


def make_form_prediction(driver_choice, circuit_choice, starting_choice, local_data, local_circuits):
    driver_code = local_data["driver_code"].loc[local_data["driverRef"] == driver_choice].values[0]
    constructor_code = local_data["constructor_code"].loc[local_data["driverRef"] == driver_choice].values[0]
    circuit_code = local_circuits["circuit_code"].loc[local_circuits["circuitRef"] == circuit_choice].values[0]
    grid_rolling = local_data["grid_rolling"].loc[local_data["driverRef"] == driver_choice].values[0]
    position_rolling = local_data["position_rolling"].loc[local_data["driverRef"] == driver_choice].values[0]
    pos_delta_rolling = local_data["pos_delta_rolling"].loc[local_data["driverRef"] == driver_choice].values[0]
    pos_delta = starting_choice
    driver_df = pd.DataFrame(
        {
            "driver_code": [driver_code],
            "constructor_code": [constructor_code],
            "circuit_code": [circuit_code],
            "grid_rolling": [grid_rolling],
            "position_rolling": [position_rolling],
            "pos_delta_rolling": [pos_delta_rolling],
            "grid": [starting_choice],
            "pos_delta": [pos_delta]
        }
    )
    driver_ds = tfdf.keras.pd_dataframe_to_tf_dataset(driver_df)
    driver_prediction = model.predict(driver_ds)
    driver_prediction_df = pd.DataFrame(driver_prediction)
    driver_full = pd.merge(driver_df, driver_prediction_df, on=driver_df.index)
    driver_full["driverRef"] = driver_choice
    driver_full["circuit_choice"] = circuit_choice
    return driver_full[["driverRef", "circuit_choice", "grid", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]


# Data prep
data = build_data("./data/final_rolling.csv")
training = data[data["year"] < 2022]
test = data[data["year"] >= 2022]
current_season = data[data["year"] == data["year"].max()]
circuits24 = pd.read_csv("./data/circuit24.csv")
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
if load_saved_model("./model/saved_model.pb") is None:
    model = build_model(training)
else:
    model = load_saved_model("./model/saved_model.pb")
inspector = model.make_inspector()
# viz = build_viz(model, training, inspector)

# Evaluator
evaluation = inspector.evaluation()
eval_perc = evaluation.accuracy * 100

# Single race table
full_table = make_prediction(model, test)
single = full_table[
    ["driverRef", "grid", "circuitRef", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
].loc[full_table["raceId"] == full_table["raceId"].max()]
single.sort_values(by="grid", inplace=True)
single["grid"] = single["grid"].astype(int)

# Data visualization
lmplot = sns.lmplot(x="pos_delta", y="grid", data=training, fit_reg=False)

# FRONTEND
st.title("Jeston Lewis - Capstone Project")
predictor, analysis = st.tabs(["Predictor", "Analysis"])


# Predictor
predictor.title("Race predictor")
predictor.write("1 - Select a driver")
predictor.write("2 - Select a track")
predictor.write("3 - Select a starting position")
predictor.write("4 - Press 'Predict'")
with predictor.form("Predict a winner"):
    f_driver_choice = st.selectbox(
        "Driver",
        current_season["driverRef"].unique(),
        index=None,
        placeholder="Choose a driver"
    )
    f_circuit_choice = st.selectbox(
        "Circuit",
        circuits24,
        index=None,
        placeholder="Choose a circuit"
    )
    f_starting_choice = st.selectbox(
        "Start",
        current_season["grid"].unique(),
        index=None,
        placeholder="Choose a starting position"
    )

    submit = st.form_submit_button("Predict")

if submit is True and f_driver_choice is not None and f_circuit_choice is not None and f_starting_choice is not None:
    prediction = make_form_prediction(
        driver_choice=f_driver_choice,
        circuit_choice=f_circuit_choice,
        starting_choice=f_starting_choice,
        local_data=current_season[current_season["raceId"] == current_season["raceId"].max()],
        local_circuits=circuits
    )
    predictor.dataframe(prediction.style.format(
        {1:"{:.2%}", 2:"{:.2%}", 3:"{:.2%}", 4:"{:.2%}", 5:"{:.2%}", 6:"{:.2%}", 7:"{:.2%}", 8:"{:.2%}",
         9:"{:.2%}", 10:"{:.2%}", 11:"{:.2%}", 12:"{:.2%}", 13:"{:.2%}", 14:"{:.2%}", 15:"{:.2%}",
         16:"{:.2%}", 17:"{:.2%}", 18:"{:.2%}", 19:"{:.2%}", 20:"{:.2%}"}
    ).highlight_max(
        axis=1,
        subset=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
        use_container_width=True,
        hide_index=True
    )
if submit is True and f_driver_choice is None or f_circuit_choice is None or f_starting_choice is None:
    predictor.write("Please select all options.")

# Analysis
analysis.title("Visualizations of the data")
analysis.header("Description of the training data")
analysis.dataframe(training.describe(), use_container_width=True)
container1 = analysis.container()
col1, col2 = container1.columns(2)
col1.header("Grid vs Position Delta")
col1.write("Starting position versus the change in position over the course of a race")
col1.pyplot(lmplot.fig)
col2.header("Header for other visual")
col2.write("")
container2 = analysis.container()
container2.title("Single race prediction")
container2.write(f"Test accuracy - {eval_perc:.2f}%")
container2.write("The table below shows the likelihood of each driver achieving a specific finishing position giving their "
           "starting position or grid. For instance, the person in first at the beginning of the race (Leclerc), has a "
           "1.67% chance of winning the race.")
container2.write("The highlighted percentage next to each driver shows the predicted likelihood of him finishing in the "
           "position indicated by the column name. Hamilton has a 83% chance of finishing in 1st.")
container2.dataframe(
    single.style.format(
        {1:"{:.2%}", 2:"{:.2%}", 3:"{:.2%}", 4:"{:.2%}", 5:"{:.2%}", 6:"{:.2%}", 7:"{:.2%}", 8:"{:.2%}",
                         9:"{:.2%}", 10:"{:.2%}", 11:"{:.2%}", 12:"{:.2%}", 13:"{:.2%}", 14:"{:.2%}", 15:"{:.2%}",
                         16:"{:.2%}", 17:"{:.2%}", 18:"{:.2%}", 19:"{:.2%}", 20:"{:.2%}"}
    ).highlight_max(
        axis=1,
        subset=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
    use_container_width=True,
    hide_index=True,
    height=738,
) # Single race prediction
container3 = analysis.container()
container3.title("Visualizations of the model")
col3, col4 = container3.columns(2)
col3.header("Single tree plot")
col3.image("tree.svg", use_column_width=True) # Plot tree
col4.header("Confusion matrix")
