from datetime import datetime

import pandas as pd
import joblib
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class PredictClustering:
    """
    This class is used to predict the clustering of new data using pre-trained models
    and a feature transformation pipeline.
    """

    script_directory = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, models_path='models'):
        """
        Initializes the PredictClustering instance and loads the feature transformation
        pipeline and clustering models from the file system.
        Parameters:
            models_path (str): The path to the directory where models and transformer are saved.
        """
        self.models_path = os.path.join(self.script_directory, '..', models_path)
        self.transformer = None
        self.models = {}

        # Load feature transformer

        transformer_path = os.path.join(models_path, 'transformer.joblib')
        #transformer_path = os.path.join(self.models_path, 'transformer.joblib')
        logging.info(f"Loading feature transformer. {transformer_path}")
        if os.path.exists(transformer_path):
            self.transformer = joblib.load(transformer_path)
        else:
            logging.error("Feature transformer file not found.")
            raise FileNotFoundError("Feature transformer file not found.")

        # Load clustering models
        logging.info("Loading clustering models.")

        for model_name in os.listdir(models_path):
            if model_name.endswith("_model.joblib") and model_name != 'transformer.joblib':
                model_path = os.path.join(models_path, model_name)
                #model_path = os.path.join(self.models_path, model_name)
                model = joblib.load(model_path)
                self.models[model_name] = model
        logging.info("All models loaded successfully.")

    def predict(self, data):
        """
        Predicts the clusters for the new data using the loaded models.
        Parameters:
            data (pd.DataFrame): The new data to cluster.
        Returns:
            dict: A dictionary containing cluster predictions from each loaded model.
        """
        if not self.transformer:
            logging.error("Transformer not loaded. Cannot proceed with prediction.")
            raise Exception("Transformer not loaded.")

        if not self.models:
            logging.error("Clustering models not loaded. Cannot proceed with prediction.")
            raise Exception("Clustering models not loaded.")

        # Transform the new data using the loaded feature transformer
        logging.info("Transforming new data.")
        _, transformed_data = self.transformer.transform(data)

        # Prepare a DataFrame to hold the predictions with CustomerID as the index
        predictions_df = pd.DataFrame(transformed_data.index).rename(columns={0: 'CustomerID'})

        for model_name, model in self.models.items():
            logging.info(f"Predicting clusters using {model_name}.")
            if hasattr(model, "predict"):
                cluster_label = model_name.replace("_model.joblib", "") + "_cluster"
                predictions_df[cluster_label] = model.predict(transformed_data)
            else:
                logging.warning(f"Model {model_name} does not support direct prediction on new data. Manual "
                                f"intervention required.")
        return predictions_df


if __name__ == "__main__":
    # Load data
    logging.info("Loading data for prediction.")
    data_df = pd.read_csv('../data/data.csv')

    # Initialize the predictor
    predictor = PredictClustering()

    # Predict clusters for the new data
    logging.info("Starting prediction.")
    cluster_predictions = predictor.predict(data_df)
    logging.info("Prediction completed.")

    # Save results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fn = f"../predictions/clustering_predictions_{timestamp}.csv"
    logging.info(f"Result successfully saved to {fn}")
    cluster_predictions.to_csv(fn, index=False)
