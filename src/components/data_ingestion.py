import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts', 'train.csv')
    test_data_path: str=os.path.join('artifacts', 'test.csv')
    raw_data_path: str=os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def clean_data(self, df:pd.DataFrame) -> pd.DataFrame:
        # Drop Duplicates
        df.drop_duplicates(inplace=True)

        # Drop Columns with too many missing values
        missing_pct = (df.isna().sum() / len(df) * 100)
        cols_to_drop = missing_pct[missing_pct > 30].index
        df.drop(columns=cols_to_drop, inplace=True)

        # Drop columns with no meaning
        df.drop(columns=['Order', 'PID'], errors='ignore', inplace=True)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        return df


    def initiate_data_ingestion(self) -> List[str]:
        logging.info('Entered the data ingestion method!')
        try:
            df=pd.read_csv(r'notebook\data\AmesHousing.csv')
            logging.info('Read the dataset as dataframe')

            df=self.clean_data(df)
            logging.info('Cleaning the dataset')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info('Train test split initiated')
            train_set, test_set=train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Ingestion of the data is completed!')

            return(
                self.ingestion_config.raw_data_path,
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)


from src.components.data_transformation import DataTransformation
if __name__=='__main__':
    ingestion_obj=DataIngestion()
    raw_path, train_path, test_path=ingestion_obj.initiate_data_ingestion()
    transform_object=DataTransformation()
    transform_object.initiate_data_transformation(raw_path, train_path, test_path)