import pandas as pd

def apply_one_hot_encoding(df, list_cols):

    for col in list_cols:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df = pd.concat([df, dummies], axis=1)
    
    df = df.drop(columns=list_cols)
        
    return df
