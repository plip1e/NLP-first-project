import pandas as pd


def clean_df(df: pd.DataFrame, drop_columns: list = ['text']) -> pd.DataFrame:

    def _is_valid_reporter(text: str) -> bool:
        words = text.split()
        prefix = text.split(' - ')[0]
        return (
            ' - ' in text and
            text.index(' - ') < len(' '.join(words[:5])) and
            not prefix.startswith('http') and
            not any(c.isdigit() for c in prefix)
        )

    def add_reporter(df: pd.DataFrame) -> pd.DataFrame: # separate reporter/agency name from article content
        for index, row in df.iterrows():
            text: str = row['text']
            if _is_valid_reporter(text):
                df.at[index, 'reporter'] = text.split(' - ')[0]
            else:
                df.at[index, 'reporter'] = "Unknown"
        return df

    def add_content(df: pd.DataFrame) -> pd.DataFrame: # separate article content from reporter/agency name
        for index, row in df.iterrows():
            text: str = row['text']
            if _is_valid_reporter(text):
                df.at[index, 'content'] = ' - '.join(text.split(' - ')[1:])
            else:
                df.at[index, 'content'] = text
        return df

    def format_date(df: pd.DataFrame) -> pd.DataFrame: # format date column and split into year, month, day
        df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
        df['year']  = df['date'].dt.year.astype('Int64')
        df['month'] = df['date'].dt.month.astype('Int64')
        df['day']   = df['date'].dt.day.astype('Int64')
        return df

    def drop_text(df: pd.DataFrame, drop_columns: list) -> pd.DataFrame:
        return df.drop(columns=drop_columns, inplace=False) # drop the original 'text' column

    def final_clean(df: pd.DataFrame) -> pd.DataFrame:
        len_before = len(df)
        df = df[df['date'].notna()]                                          # drop rows with unparseable date
        df = df[df['content'].notna() & (df['content'].str.strip() != '')]  # drop rows with missing/empty content
        df = df[~df['content'].str.startswith(('http://', 'https://'))]     # drop rows where content is just a URL
        df = df.reset_index(drop=True)
        df.drop(columns='date')
        print(f"Rows dropped: {len_before - len(df)}")
        return df

    df = add_reporter(df)
    df = add_content(df)
    df = format_date(df)
    df = final_clean(df)
    df = drop_text(df, drop_columns)
    return df

