import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import pandas as pd
import pickle
import ruptures as rpt

data= pd.read_csv(r"timeseries/dataset_predictions/dipteryx_sgbt.csv")


with open(r'timeseries/models/xgb_model.pkl', 'rb') as file:
      model = pickle.load(file)

X=data[['rccM', 'gccM', 'bccM', 'ExGM', 'gvM', 'npvM', 'shadowM','rSD', 'gSD', 'bSD',
       'ExGSD', 'gvSD', 'npvSD', 'gcorSD', 'gcorMD','entropy','elevSD']]

Y= data[['area', 'score', 'tag', 'GlobalID', 'iou',
       'date', 'latin', 'polygon_id']]
X_predicted=model.predict(X)
df_final = Y.copy()  # Copy Y to keep the same structure
df_final['leafing_predicted'] = X_predicted

df_final['date'] = pd.to_datetime(df_final['date'], format='%Y_%m_%d')
df_final['dayYear'] = df_final['date'].dt.dayofyear
df_final['year']= df_final['date'].dt.year
df_final['date_num']= (df_final['date'] -df_final['date'].min()).dt.days

individuals=pd.unique(df_final['GlobalID'])
indv= df_final[df_final['GlobalID']==individuals[18]]

indv=indv[['leafing_predicted', 'dayYear','year','date_num','date']]

indv = indv.sort_values(by='date_num')
full_date_range = pd.date_range(start=indv['date'].min(), end=indv['date'].max(), freq='D')
full_df = pd.DataFrame({'date': full_date_range})
full_df['dayYear'] = full_df['date'].dt.dayofyear  # Assuming 'date_num' corresponds to day of year
full_df['date_num']= (full_df['date']- full_df['date'].min()).dt.days
full_df['year'] = full_df['date'].dt.year  # Extract year if needed

# Merge the full_df with the original dataframe to identify missing dates
indv = pd.merge(full_df, indv, on=['date', 'date_num', 'year','dayYear'], how='left')
indv['leafing_predicted'] = indv['leafing_predicted'].interpolate(method='linear')


plt.figure(figsize=(12, 6))
plt.scatter(indv['dayYear'], indv['leafing_predicted'], c=indv['year'], cmap='viridis', label='Leafing Predicted')
plt.xlabel('Day of year')
plt.ylabel('Leafing Predicted')
plt.grid(True)
plt.title('Leafing Predicted vs Day of Year')
plt.legend()
plt.colorbar(label='Year')
plt.show()



# Extract the signal
signal = indv['leafing_predicted'].values  # Get the leafing predictions

# Apply rupture's dynamic programming method
algo = rpt.Dynp(model="l2").fit(signal)

# Predict the change points
n_bkps = 11  # Number of change points to detect
result = algo.predict(n_bkps=n_bkps)

# Show the change points
print("Change points detected at indices:", result)

rpt.display(signal, result)
plt.show()

# Plot the leafing predictions
plt.figure(figsize=(12, 6))
plt.scatter(indv['date_num'], indv['leafing_predicted'], c=indv['year'], cmap='viridis', label='Leafing Predicted')
plt.xlabel('Day of year')
plt.ylabel('Leafing Predicted')
plt.grid(True)
plt.title('Leafing Predicted vs Day of Year')
plt.legend()
plt.colorbar(label='Year')

# Add the detected breakpoints as dashed vertical lines
for bp in result:
    plt.axvline(x=bp, color='red', linestyle='--', linewidth=2)

# Show the plot
plt.show()