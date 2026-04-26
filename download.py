import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def download_data():
    url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
    df = pd.read_csv(url)
    df.to_csv("insurance_raw.csv", index=False)
    return df

def clear_data(path2df):
    df = pd.read_csv(path2df)
    df = df.dropna()
    cat_columns = ['sex', 'smoker', 'region']
    ordinal = OrdinalEncoder()
    ordinal.fit(df[cat_columns])
    df[cat_columns] = ordinal.transform(df[cat_columns])
    df = df[df['bmi'] < 60]
    df = df[df['charges'] < 60000]
    df.to_csv('df_clear.csv', index=False)
    return True

if __name__ == "__main__":
    download_data()
    clear_data("insurance_raw.csv")
