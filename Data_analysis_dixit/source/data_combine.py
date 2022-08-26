import os
import pandas as pd

def data_gathering(file_path):
    frames = []
    for name in os.listdir(file_path):
        extents = os.path.splitext(file_path+'/{}'.format(name))[1].lower()
        if extents == '.csv':
            df= pd.read_csv(file_path+'/{}'.format(name))
            frames.append(df)
        elif extents == '.xlsx':
            df= pd.read_excel(file_path+'/{}'.format(name))
            df.columns = df.columns.astype(str)
            frames.append(df)
    result =  pd.concat(frames, axis=0, ignore_index= True)

    return result