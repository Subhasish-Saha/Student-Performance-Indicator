import os
import sys
import pickle
import numpy as np
import pandas as pd

from src.exception import Custom_Exception

def save_object(file_path:str, obj:object) -> None:
    '''
    This funciton will save the given object in the given file path.
    '''
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok= True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
    
    except Exception as e:
        raise Custom_Exception(e,sys)