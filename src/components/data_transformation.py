import os
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from typing import List
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts', 'preprocessor.pkl')



class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_numerical_categorical_cols(self, raw_path:str) -> List:
        df=pd.read_csv(raw_path)
        num_cols=df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols=df.select_dtypes(include=['object', 'category']).columns.tolist()

        return(
            num_cols,
            cat_cols
        )

    def get_data_transformer_object(self, raw_path:str) -> ColumnTransformer:
        '''
        This function is responsible for data transformation
        '''
        try:
            num_cols, cat_cols=self.get_numerical_categorical_cols(raw_path)
            target_column_name='SalePrice'
            unscaled_cols = ['Year Built', 'Year Remod/Add', 'Garage Yr Blt']

            num_cols = [
                col for col in num_cols
                if col != target_column_name and col not in unscaled_cols
            ]

            cat_cols = [col for col in cat_cols if col != target_column_name]

            num_pipeline=Pipeline(
                steps=[
                    ('Imputer', SimpleImputer(strategy='median')),
                    ('Scaler', StandardScaler())
                ]
            )

            logging.info('Numerical Columns Scaling Pipeline Completed')

            cat_pipeline=Pipeline(
                steps=[
                    ('Imputer', SimpleImputer(strategy='most_frequent')),
                    ('One_Hot_Encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ]
            )

            logging.info('Categorical Columns Encoding Pipeline Completed')

            unscaled_pipeline=Pipeline(
                steps=[
                    ('Imputer', SimpleImputer(strategy='median'))
                ]
            )

            logging.info('Unscaled Columns Scaling Pipeline Completed')


            preprcoessor=ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, num_cols),
                    ('cat_pipeline', cat_pipeline, cat_cols),
                    ('unscaled_pipeline', unscaled_pipeline, unscaled_cols)
                ]
            )

            return preprcoessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, raw_path:str, train_path:str, test_path:str) -> List[str]:
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info('Obtaining Preprocessor Object')

            preprocessing_obj=self.get_data_transformer_object(raw_path)
            target_column_name='SalePrice'

            target_feature_train=train_df[target_column_name]
            target_feature_test=test_df[target_column_name]

            input_feature_train=train_df.drop(columns=[target_column_name], axis=1)
            input_feature_test=test_df.drop(columns=[target_column_name], axis=1)

            logging.info('Applying preprocessing on training dataframe and testing dataframe.')

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test)
            ]

            logging.info('Saving preprocessing object.')

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)
