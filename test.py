import pandas as pd
chronos = pd.read_csv('outputs/predictions/chronos_fintext_INDX.SPX.csv')
print('Chronos predictions:')
print(chronos[['datetime', 'actual', 'predicted', 'predicted_calib']].head(10))
print(chronos['predicted'].describe())