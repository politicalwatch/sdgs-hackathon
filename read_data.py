import pandas as pd

def json_to_df(file_path,columns_to_keep=["initiative_type","initiative_type_alt","content","topics","title","content"]):
    df = pd.read_json(file_path)
    subset = df[columns_to_keep]
    return subset

initiative_data = json_to_df('./initiatives2.json')