import pandas as pd

def download_data(url: str, output_path: str):
    df = pd.read_csv(url)
    df.to_csv(output_path, index=False)
    return df

if __name__ == "__main__":
    DATA_URL = 'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv'
    download_data(DATA_URL, 'insurance_raw.csv')
