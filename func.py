import pandas as pd


def clean_df(df: pd.DataFrame, drop_columns: list = ['text', 'date']) -> pd.DataFrame:

    def add_reporter(df: pd.DataFrame) -> pd.DataFrame: # seporate reporter name from article content
        for index, row in df.iterrows():
            text: str = row['text']
            words = text.split()
            # check if ' - ' is within the first 5 words
            if ' - ' in text and text.index(' - ') < len(' '.join(words[:5])):
                df.at[index, 'reporter'] = text.split(' - ')[0]
            else: df.at[index, 'reporter'] = "Unknown"
        return df
    
    def add_content(df: pd.DataFrame) -> pd.DataFrame: # seporate article content from reporter name
        for index, row in df.iterrows():
            text: str = row['text']
            words = text.split()
            # check if ' - ' is within the first 5 words
            if ' - ' in text and text.index(' - ') < len(' '.join(words[:5])):
                df.at[index, 'content'] = text.split(' - ')[1]
            else: df.at[index, 'content'] = text
        return df
    
    def format_date(df: pd.DataFrame) -> pd.DataFrame: # format the date column from (January 15 2023) to datetime format and split to 3 columns (year, month, day)
        mask = pd.to_datetime(df['date'], format='mixed', errors='coerce').isna()
        print(df[mask]['date'].value_counts().head(20))
        df['date'] = pd.to_datetime(df['date'], format='%B %d, %Y', errors='coerce') # convert to datetime format, if error occurs, set to NaT
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        return df

    def final_clean(df: pd.DataFrame) -> pd.DataFrame:
        len_before = len(df)
        df = df[df['date'].notna()]
        df = df[df['content'].notna() & (df['content'].str.strip() != '')]  # catches empty strings too
        df = df.reset_index(drop=True)  # clean up index after drops
        print(f"Rows dropped: {len_before - len(df)}/{len_before} ({(len_before - len(df)) / len_before:.2%})")
        return df
    
    def drop_text(df: pd.DataFrame, drop_columns: list) -> pd.DataFrame: 
        return df.drop(columns=drop_columns, inplace=False) # drop the original 'text' column
        
    df = add_reporter(df)
    df = add_content(df)
    df = format_date(df)
    df = drop_text(df, drop_columns)
    df = final_clean(df)
    return df