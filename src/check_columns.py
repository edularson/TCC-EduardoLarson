# Adicione este script: check_columns.py
import pandas as pd
df = pd.read_csv('data/statsbomb_shots_laliga.csv')
print(df.columns.tolist())
print(df.head(2).to_string())