import os
import sys
import pickle
from src.exception import CustomException


def save_object(file_path: str, obj: object):
    """
    Saves a Python object (e.g., ColumnTransformer, Model) to a pickle file.
    Creates parent directories automatically if they do not exist.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)