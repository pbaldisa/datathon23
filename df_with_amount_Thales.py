import pandas as pd

def units_for_next_interval(days_to_next_interval, days_to_next_purchase, amount):
    # use thales theorem to estimate how many units need to be sent on the next interval
    return int((amount*(days_to_next_purchase-days_to_next_interval))/(days_to_next_purchase))

# loading files
df = pd.read_csv('processed_df.csv')

# Features that we are not using
col_to_del = ['ORIGEN','IMPORTELINEA','PRECIO','REFERENCIA','PRODUCTO']
df.drop(columns=col_to_del, inplace=True, axis=1)

df['FECHAPEDIDO'] = pd.to_datetime(df['FECHAPEDIDO'])
#df['FormattedDate'] = df['FECHAPEDIDO'].dt.strftime('%Y%m%d')
#df.sort_values(by='FormattedDate', inplace=True)
df['FECHAPEDIDO'] = df['FECHAPEDIDO'].dt.dayofyear
df['global_day'] = (df['year']-15)*365 + df['FECHAPEDIDO']
df.sort_values(by='global_day', inplace=True)

df['amount'] = df['CANTIDADCOMPRA']*df['UNIDADESCONSUMOCONTENIDAS']

codigo_to_filter = 'B41691'
hospital_to_filter = 10
filtered_df = df[(df['CODIGO'] == codigo_to_filter) & (df['id_hospital'] == hospital_to_filter)]

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
print(filtered_df.shape[0])


'''scaler = MinMaxScaler(feature_range=(0,1))
# scaling dataset
scaled_train = scaler.fit_transform(dataset_train)

print(scaled_train[:5])
# Normalizing values between 0 and 1
scaled_test = scaler.fit_transform(dataset_test)
print(*scaled_test[:5]) #prints the first 5 rows of scaled_test

# finally, we normalize our training data
# scaler = StandardScaler()
scaler = MinMaxScaler()
dataset_train = scaler.fit_transform(dataset_train)
dataset_test = scaler.transform(dataset_test)'''

