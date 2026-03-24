import pandas as pd


def clean_df(df: pd.DataFrame, drop_columns: list = ['text']) -> pd.DataFrame:

    def add_reporter(df: pd.DataFrame) -> pd.DataFrame: # seporate reporter name from article content
        clean_text = ''
        for index, row in df.iterrows():
            text = row['text']
            text = clean_text(text)
            words = text.split()
            # check if ' - ' is within the first 5 words
        if ' - ' in text and text.index(' - ') < len(' '.join(words[:5])):
            df.at[index, 'reporter'] = text.split(' - ')[0]
        else: df.at[index, 'reporter'] = "Unknown"
        return df
    
    def add_content(df: pd.DataFrame) -> pd.DataFrame: # seporate article content from reporter name
        for index, row in df.iterrows():
            text = row['text']
            words = text.split()
            # check if ' - ' is within the first 5 words
        if ' - ' in text and text.index(' - ') < len(' '.join(words[:5])):
            df.at[index, 'content'] = text.split(' - ')[1]
        else: df.at[index, 'content'] = text
        return df
    
    def drop_text(df: pd.DataFrame, drop_columns: list) -> pd.DataFrame: 
        df = df.drop(columns=[drop_columns]) # drop the original 'text' column
        return df
    
    df = add_reporter(df)
    df = add_content(df)
    df = drop_text(df, drop_columns)
    return df