import logging
import pandas as pd
import joblib
import os
from sklearn.cluster import KMeans, AgglomerativeClustering
from src.features_transformer import FeaturesTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class TrainClusteringModel:
    """
    This class is responsible for the entire pipeline of feature transformation and
    training clustering models on the provided dataset.
    """
    def __init__(self, data_path):
        """
        Initializes the TrainClusteringModel instance.
        Parameters:
            data_path (str): The path to the dataset file.
        """
        self.data_path = data_path
        self.transformer = FeaturesTransformer()  # Initialize the transformer here
        self.models = {}

    def load_and_transform_data(self):
        """
        Loads the data from the given path and applies feature transformation.
        """
        logging.info("Loading data.")
        df = pd.read_csv(self.data_path)
        logging.info("Data loaded successfully.")

        logging.info("Starting feature transformation.")
        df_transformed, customer_df = self.transformer.fit_transform(df)
        logging.info("Feature transformation complete.")
        return df_transformed, customer_df

    def fit(self):
        """
        Trains clustering models on the transformed dataset.
        """
        # Load and transform the data
        df_transformed, customer_df = self.load_and_transform_data()

        logging.info("Starting the fitting of clustering models.")
        # Loop over the number of clusters for KMeans and Agglomerative
        for n_clusters in [3, 4, 5]:
            logging.info(f"Training KMeans with n_clusters={n_clusters}")
            # KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(customer_df)
            self.models[f'kmeans_{n_clusters}'] = kmeans
        logging.info("Finished fitting all clustering models.")

    def save_models_and_transformer(self):
        """
        Saves the trained clustering models and the feature transformation pipeline.
        """
        # Ensure the models directory exists
        os.makedirs('../models', exist_ok=True)

        # Save the transformation pipeline
        joblib.dump(self.transformer, '../models/transformer.joblib')
        logging.info("Feature transformer saved successfully.")

        # Save clustering models
        for name, model in self.models.items():
            joblib.dump(model, f'../models/{name}_model.joblib')
        logging.info("All models saved successfully.")


if __name__ == "__main__":
    train_file_path = '../data/data.csv'
    trainer = TrainClusteringModel(train_file_path)

    logging.info("Initiating model training.")
    trainer.fit()  # Fit all models including data transformation

    logging.info("Saving trained models and transformer.")
    trainer.save_models_and_transformer()  # Save all models and the transformer

    logging.info("Training and saving process is DONE.")
