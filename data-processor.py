import pandas as pd

df = pd.read_csv('./parkinsons-data.csv')

classification = ['Class']
features = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22']

df = df.sample(frac=1).reset_index(drop=True)

df[features].to_csv('./X.csv', index=False)
df[classification].to_csv('./Y.csv', index=False)