import shutil
import os
import sys
import pandas as pd
import pickle
from src.logger import logging
from src.exception import CustomException
from flask import request
from src.constant import *
from src.utils.main_utils import MainUtils
from dataclasses import dataclass


@dataclass
class PredictionPipelineConfig:
    prediction_output_dirname: str = 'predictions'
    prediction_file_name: str = 'prediction_file_csv'
    model_file_path: str = os.path.join(artifact_folder, 'model.pkl')
    preprocessor_path: str = os.path.join(artifact_folder, 'preprocessor.pkl')
    prediction_file_path: str = os.path.join(
        prediction_output_dirname,
        prediction_file_name
    )


class PredictionPipeline:

    def __init__(self, request: request):

        self.request = request
        self.utils = MainUtils()
        self.prediction_pipeline_config = PredictionPipelineConfig()


    def save_input_files(self) -> str:

        try:

            pred_file_input_dir = 'prediction_artifacts'

            os.makedirs(
                pred_file_input_dir,
                exist_ok=True
            )

            input_csv_file = self.request.files['file']

            if input_csv_file.filename == '':
                raise Exception("Please select a CSV file")

            pred_file_path = os.path.join(
                pred_file_input_dir,
                input_csv_file.filename
            )

            input_csv_file.save(pred_file_path)

            return pred_file_path

        except Exception as e:

            raise CustomException(e, sys)


    def predict(self, features):

        try:

            print(
                "Prediction features:",
                features.columns.tolist()
            )

            model = self.utils.load_object(
                self.prediction_pipeline_config.model_file_path
            )

            preprocessor = self.utils.load_object(
                file_path=self.prediction_pipeline_config.preprocessor_path
            )

            transformed_x = preprocessor.transform(features)

            preds = model.predict(transformed_x)

            return preds

        except Exception as e:

            raise CustomException(e, sys)


    def get_predicted_dataframe(
        self,
        input_dataframe_path: pd.DataFrame
    ):

        try:

            prediction_column_name: str = TARGET_COLUMN

            input_dataframe: pd.DataFrame = pd.read_csv(
                input_dataframe_path
            )

            # Remove unwanted index column
            if "Unnamed: 0" in input_dataframe.columns:

                input_dataframe = input_dataframe.drop(
                    columns=["Unnamed: 0"]
                )

            # Remove target column from prediction input
            if "Good/Bad" in input_dataframe.columns:

                input_dataframe = input_dataframe.drop(
                    columns=["Good/Bad"]
                )

            # Remove TARGET_COLUMN also if it is different
            if (
                TARGET_COLUMN in input_dataframe.columns
                and TARGET_COLUMN != "Good/Bad"
            ):

                input_dataframe = input_dataframe.drop(
                    columns=[TARGET_COLUMN]
                )

            # Load preprocessor
            preprocessor = self.utils.load_object(
                file_path=self.prediction_pipeline_config.preprocessor_path
            )

            # Make sure prediction columns match training columns
            if hasattr(preprocessor, "feature_names_in_"):

                trained_features = list(
                    preprocessor.feature_names_in_
                )

                input_dataframe = input_dataframe[
                    trained_features
                ]

            print(
                "Final prediction columns:",
                input_dataframe.columns.tolist()
            )

            # Prediction
            prediction = self.predict(
                input_dataframe
            )

            # Add prediction column
            input_dataframe[prediction_column_name] = [
                pred for pred in prediction
            ]

            # Convert prediction values
            target_column_mapping = {
                0: ' bad',
                1: 'good'
            }

            input_dataframe[prediction_column_name] = (
                input_dataframe[prediction_column_name]
                .map(target_column_mapping)
            )

            # Create prediction output directory
            os.makedirs(
                self.prediction_pipeline_config.prediction_output_dirname,
                exist_ok=True
            )

            # Save prediction CSV
            input_dataframe.to_csv(
                self.prediction_pipeline_config.prediction_file_path,
                index=False
            )

            logging.info("predictions completed")

        except Exception as e:

            raise CustomException(e, sys) from e


    def run_pipeline(self):

        try:

            input_csv_path = self.save_input_files()

            self.get_predicted_dataframe(
                input_csv_path
            )

            return self.prediction_pipeline_config

        except Exception as e:

            raise CustomException(e, sys)