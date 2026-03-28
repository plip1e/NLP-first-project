import pandas as pd
import numpy as np
import re, string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from sklearn.preprocessing import OneHotEncoder


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

def tokeniseEmbed_and_oneHot(df: pd.DataFrame, tokenised_columns: list[str], oneHotEnc_columns: list[str], test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:

    def remove_punctuation(tokenised_text: str) -> list[str]:
        pattern = re.compile('[%s]' % re.escape(string.punctuation))
        new_sentence = []
        for token in tokenised_text:
            new_token = pattern.sub(u'', token)
            if not new_token == u'':
                new_sentence.append(new_token)
        return new_sentence

    def tokenise_columns(df: pd.DataFrame, tokenised_columns: list[str]) -> pd.DataFrame:
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        for column in tokenised_columns:
            temp_col = []
            for text in df[column].values:
                filtered_tokens = []
                tokenised_text = remove_punctuation(word_tokenize(text))
                for token in tokenised_text:
                    if token.lower() not in stop_words:
                        filtered_tokens.append(token.lower())
                lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
                temp_col.append(lemmatized_tokens)
            df[column] = pd.Series(temp_col)
        return df

    def embed_tokenised_text(df_train: pd.DataFrame, df_test: pd.DataFrame, tokenised_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        for column in tokenised_columns:
            # train Word2Vec on training data only — no leakage
            model = Word2Vec(sentences=df_train[column].values, vector_size=100, window=5, min_count=1, workers=4)
            for df in [df_train, df_test]:
                df[column] = df[column].apply(
                    lambda tokens: np.mean([model.wv[token] for token in tokens if token in model.wv], axis=0)
                    if tokens else np.zeros(100)
                )
        return df_train, df_test

    def oneHotEncode_columns(df: pd.DataFrame, oneHotEnc_columns: list[str]) -> pd.DataFrame:
        encoder = OneHotEncoder(sparse_output=False)
        for column in oneHotEnc_columns:
            encoded_data = encoder.fit_transform(df[[column]])
            encoded_df = pd.DataFrame(encoded_data, columns=[f"{column}_{cat}" for cat in encoder.categories_[0]])
            df = pd.concat([df, encoded_df], axis=1).drop(column, axis=1)
        return df

    # tokenise first on full df (no leakage risk, it's just text cleaning)
    df = tokenise_columns(df, tokenised_columns)

    # split before embedding
    df_train = df.sample(frac=1 - test_size, random_state=42)
    df_test  = df.drop(df_train.index)

    # embed using train-only Word2Vec
    df_train, df_test = embed_tokenised_text(df_train, df_test, tokenised_columns)

    # OHE applied to both separately but fit on train only
    if oneHotEnc_columns:
        encoder = OneHotEncoder(sparse_output=False)
        for column in oneHotEnc_columns:
            encoder.fit(df_train[[column]])
            for split in [df_train, df_test]:
                encoded_data = encoder.transform(split[[column]])
                encoded_df = pd.DataFrame(encoded_data, columns=[f"{column}_{cat}" for cat in encoder.categories_[0]], index=split.index)
                split.drop(columns=column, inplace=True)
                split[encoded_df.columns] = encoded_df

    return df_train, df_test