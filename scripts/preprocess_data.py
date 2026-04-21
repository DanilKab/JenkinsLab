import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def clean_and_encode(input_path: str, output_path: str):
    df = pd.read_csv(input_path)

    df = df[(df['bmi'] >= 15) & (df['bmi'] <= 50)]
    
    df = df[df['age'] <= 100]
    
    df = df[df['charges'] > 0]

    df = df.reset_index(drop=True)

    cat_columns = ['sex', 'smoker', 'region']
    ordinal_encoder = OrdinalEncoder()
    df[cat_columns] = ordinal_encoder.fit_transform(df[cat_columns])

    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    clean_and_encode('insurance_raw.csv', 'insurance_processed.csv')
