import os
import pandas as pd

def create_folder_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def convert_datetime(df, list_of_col):
  def convert_datetime_by_row(row):
    try:
      return pd.to_datetime(row, format='%d-%b-%Y',errors='raise').date()
    except:
      return pd.to_datetime(row, format='%Y-%m-%dT%H:%M:%SZ',errors='raise').date()

  result = df.copy()
  for col in list_of_col:
    result[col] =   result[col].apply(convert_datetime_by_row)

  return result