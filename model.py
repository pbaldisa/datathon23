import pandas as pd
import numpy as np
import statsmodels.api as sm

def greedy(df, exp_date):
    paquet_sizes = np.unique(df['UNIDADESCONSUMOCONTENIDAS'])
    paquet_sizes = sorted(paquet_sizes)

    clean_df = filtered_df.copy()

    clean_df.sort_values(by='global_day', inplace=True)
    clean_df['total_units'] = clean_df['total_units'].astype(int)

    for index, row in clean_df.iterrows():  
        if row['total_units']!=0:
            for i in range(1,exp_date+1):
                if index+i<len(clean_df):
                    total_units = clean_df.loc[clean_df.index[index+i],'total_units']
                    if total_units!=0:
                        row['total_units'] += total_units
                        clean_df.loc[clean_df.index[index+i],'total_units'] = 0

            # set the apropiate packet size with a greedy approach
            i_max = len(paquet_sizes)-1
            # find the biggest package that's lower than the total units
            while row['total_units'] < paquet_sizes[i_max] and i_max>=0:
                i_max-=1

            # now we can assume row['total_unit'] >= paquet_sizes[i_max]
            # we need to fill the last package
            row['total_units']+= (row['total_units'] // paquet_sizes[i_max])

    return clean_df

def units_for_next_interval(days_to_next_interval, days_to_next_purchase, amount):
    # use thales theorem to estimate how many units need to be sent on the next interval
    return int((amount*(days_to_next_purchase-days_to_next_interval))/(days_to_next_purchase))

# loading files
df = pd.read_csv('processed_df.csv')

# Features that we are not using
# One hot encode 'TIPOCOMPRA', 'TGL' 
df = pd.get_dummies(df, columns=['TIPOCOMPRA', 'TGL'], prefix=['TIPOCOMPRA', 'TGL'], dtype='int64')

col_to_del = ['ORIGEN','IMPORTELINEA','REFERENCIA','PRODUCTO']
df.drop(columns=col_to_del, inplace=True, axis=1)

df['FECHAPEDIDO'] = pd.to_datetime(df['FECHAPEDIDO'])
#df['FormattedDate'] = df['FECHAPEDIDO'].dt.strftime('%Y%m%d')
#df.sort_values(by='FormattedDate', inplace=True)

df['global_day'] = (df['year']-15)*365 + df['FECHAPEDIDO'].dt.dayofyear
df.sort_values(by='global_day', inplace=True)

df['amount'] = df['CANTIDADCOMPRA']*df['UNIDADESCONSUMOCONTENIDAS']

codigo_to_filter = 'B41691'
hospital_to_filter = 10
filtered_df = df[(df['CODIGO'] == codigo_to_filter) & (df['id_hospital'] == hospital_to_filter)]

caducitat = int(filtered_df['Caducidad'].iloc[-1])

###############
# CLEAN DATASET using Thales
###############
day = 1
length_interval = 5 # days
columns = filtered_df.columns
clean_df = pd.DataFrame(columns=columns)
clean_df = clean_df.iloc[0:0].copy()

# Get the first row
first_row = filtered_df.iloc[0]
first_day = first_row['global_day']

filtered_df = filtered_df.reset_index(drop=True)

while first_day >= day + length_interval:
    day += length_interval

clean_df = pd.concat([clean_df, pd.DataFrame([first_row], columns=columns)], ignore_index=True)
clean_df.loc[clean_df.index[-1], 'global_day'] = day
clean_df.loc[clean_df.index[-1], 'amount'] = 0

# Using vectorized operations
for index, row in filtered_df.iterrows():  
    # we can assume day <= row['global_day'] < day + length_interval


    if index+1 < filtered_df.shape[0]: # ignore the last command as we have no info of the next one
        next_row = filtered_df.iloc[index+1]
        x = units_for_next_interval(day + length_interval - row['global_day'], next_row['global_day'] - row['global_day'], row['amount'])

        # Append the new row and override the original DataFrame
        #print('inserting row:')
        clean_df.loc[clean_df.index[-1], 'amount'] += (row['amount'] - x)

        remaining_amount = x
        day += length_interval

        while next_row['global_day'] >= day + length_interval:
            
            x = units_for_next_interval(length_interval, next_row['global_day'] - day, remaining_amount)

            # Append the new row and override the original DataFrame
            clean_df = pd.concat([clean_df, pd.DataFrame([row], columns=columns)], ignore_index=True)
            clean_df.loc[clean_df.index[-1], 'amount'] = remaining_amount - x
            clean_df.loc[clean_df.index[-1], 'global_day'] = day

            remaining_amount = x
            day += length_interval

        # Now we can assume day <= next_row['global_day'] < day + length_interval for next iteration
        clean_df = pd.concat([clean_df, pd.DataFrame([row], columns=columns)], ignore_index=True)
        clean_df.loc[clean_df.index[-1],'amount'] = remaining_amount
        clean_df.loc[clean_df.index[-1],'global_day'] = day


print('################################################333')
print(clean_df[['amount','global_day']])

clean_df = clean_df.dropna()


# Create train test split
train_df = clean_df[clean_df['FECHAPEDIDO'] < '2023-01-01'].copy()
test_df = clean_df[clean_df['FECHAPEDIDO'] >= '2023-01-01'].copy()

train_df.drop(columns=['FECHAPEDIDO','CODIGO','NUMERO'], inplace=True, axis=1)
test_df.drop(columns=['FECHAPEDIDO','CODIGO','NUMERO'], inplace=True, axis=1)

print(train_df)

# Adding exogenous features
exogenous_features = [-4, -3, -6, -5, 2]
# Create a list of hospital id for hospitals with at least 100 orders
hospital_order_counts = clean_df.groupby('id_hospital').size()
big_hospitals = hospital_order_counts[hospital_order_counts >= 100].index.tolist()

models_by_hospital = dict()


# Train an SARIMAX model for each combination of 'id_hospital' and product code ('CODIGO')
# for hospital in big_hospitals[0]:
hospital=big_hospitals[0]
# Filter rows where 'id_hospital' is 'hospital' and 'CODIGO' is 'codigo'
subset = train_df[(train_df['id_hospital'] == hospital)]
subset = np.asarray(subset)
subset = np.array(subset, dtype=float)
# Create a SARIMAX model
model = sm.tsa.statespace.SARIMAX(subset[:,-1], exog=subset[:,exogenous_features], order=(6, 0, 6), seasonal_order=(1, 1, 1, 365/length_interval))

# Fit the model
model_fit = model.fit()

# Add the model to the dictionary
models_by_hospital[hospital] = model_fit

# print(greedy(clean_df, caducitat))




