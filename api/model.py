import os
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_decision_forests as tfdf

from api.data_pipeline import PREDICTORS, POSITION_COLUMNS
from api.database import SessionLocal, ModelRun

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "model"))


class F1Model:
    """Manages the TensorFlow Decision Forests model lifecycle."""

    def __init__(self):
        self.model = None
        self.inspector = None
        self.accuracy = None

    def load(self) -> bool:
        """Load a saved model from disk. Returns True if successful."""
        saved_path = os.path.join(MODEL_DIR, "saved_model.pb")
        if os.path.exists(saved_path):
            try:
                self.model = tf.keras.models.load_model(MODEL_DIR)
                self.inspector = self.model.make_inspector()
                self.accuracy = self.inspector.evaluation().accuracy
                return True
            except Exception:
                return False
        return False

    def train(self, training_data: pd.DataFrame) -> float:
        """Train a new model on the given data. Returns accuracy."""
        train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
            training_data[PREDICTORS], label="position"
        )
        self.model = tfdf.keras.RandomForestModel()
        self.model.fit(train_ds)

        os.makedirs(MODEL_DIR, exist_ok=True)
        self.model.save(MODEL_DIR)

        self.inspector = self.model.make_inspector()
        self.accuracy = self.inspector.evaluation().accuracy

        # Log training run
        session = SessionLocal()
        try:
            run = ModelRun(
                trained_at=datetime.datetime.now(),
                accuracy=self.accuracy,
                num_samples=len(training_data),
            )
            session.add(run)
            session.commit()
        finally:
            session.close()

        return self.accuracy

    def predict_single(self, driver_df: pd.DataFrame) -> pd.DataFrame:
        """Make prediction for a single driver scenario. Returns DataFrame with probabilities."""
        ds = tfdf.keras.pd_dataframe_to_tf_dataset(driver_df)
        predictions = self.model.predict(ds)
        pred_df = pd.DataFrame(predictions, columns=POSITION_COLUMNS)
        result = pd.concat([driver_df.reset_index(drop=True), pred_df], axis=1)
        return result

    def predict_batch(self, data: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """Make predictions for multiple rows. Returns merged table and raw predictions."""
        ds = tfdf.keras.pd_dataframe_to_tf_dataset(data[PREDICTORS])
        predictions = self.model.predict(ds)
        pred_df = pd.DataFrame(predictions, columns=POSITION_COLUMNS)
        table = pd.merge(data.reset_index(drop=True), pred_df, left_index=True, right_index=True)
        return table, predictions

    def get_confusion_matrix(self, test_data: pd.DataFrame) -> list[list[int]]:
        """Compute confusion matrix for test data."""
        _, predictions = self.predict_batch(test_data)
        one_prediction = tf.argmax(predictions, axis=-1)
        cm = tf.math.confusion_matrix(labels=test_data["position"].values, predictions=one_prediction)
        return cm.numpy().tolist()


# Global singleton
f1_model = F1Model()
