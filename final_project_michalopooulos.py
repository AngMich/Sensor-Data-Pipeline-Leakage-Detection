

import os
import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier

"""**reading from directory and creating all_data files**"""

sensor_match_1 = re.search(r'Sensor([12])', 'data/ITC4568 Final_Project_data/Sensor1_Leakage_2022-12-09_10-45-36')
sensor_id = sensor_match_1.group(1)
output_name = f"ALL_DATA_{sensor_id}.CSV"

all_data_1 = []

for root, dirs, files in os.walk('data/ITC4568 Final_Project_data/Sensor1_Leakage_2022-12-09_10-45-36'):
    for file in files:
        if file.endswith(".csv"):
            root_folder = os.path.basename(root)
            parts = root_folder.split('_')
            if len(parts) >= 2:
                option = parts[1]
                if option == 'AllTapsClosed':
                    label = 0
                elif option == 'Tap1Open':
                    label = 1
                elif option == 'Tap2Open':
                    label = 2
                elif option == 'Tap3Open':
                    label = 3
                elif option == 'Tap4Open':
                    label = 4
                if label is not None:
                    file_path = os.path.join(root, file)
                    df = pd.read_csv(file_path)
                    df['label'] = label
                    all_data_1.append(df)
if not all_data_1:
    print("No data found")

final_df_1 = pd.concat(all_data_1)

final_df_1.sort_values(by='Time', inplace=True)

final_df_1.to_csv(output_name, index=False)
print('success created: ', output_name)

sensor_match_2 = re.search(r'Sensor([12])', 'data/ITC4568 Final_Project_data/Sensor2_Leakage_2022-12-09_10-46-58')
sensor_id = sensor_match_2.group(1)
output_name = f"ALL_DATA_{sensor_id}.CSV"

all_data_2 = []

for root, dirs, files in os.walk('data/ITC4568 Final_Project_data/Sensor2_Leakage_2022-12-09_10-46-58'):
    for file in files:
        if file.endswith(".csv"):
            root_folder = os.path.basename(root)
            parts = root_folder.split('_')
            if len(parts) >= 2:
                option = parts[1]
                if option == 'AllTapsClosed':
                    label = 0
                elif option == 'Tap1Open':
                    label = 1
                elif option == 'Tap2Open':
                    label = 2
                elif option == 'Tap3Open':
                    label = 3
                elif option == 'Tap4Open':
                    label = 4
                if label is not None:
                    file_path = os.path.join(root, file)
                    df = pd.read_csv(file_path)
                    df['label'] = label
                    all_data_2.append(df)
if not all_data_2:
    print("No data found")

final_df_2 = pd.concat(all_data_2)

final_df_2.sort_values(by='Time', inplace=True)

final_df_2.to_csv(output_name, index=False)
print('success created: ', output_name)

"""**creating aggregated files for data 1 and data 2**"""

df1 = pd.read_csv('ALL_DATA_1.csv')

df1['ts'] = (df1['Time']//0.1)*0.1
df1['ts'] = df1['ts'].round(1)

df1['group'] = (df1['label'] != df1['label'].shift()).cumsum()

agg1 = df1.groupby(['ts', 'group','label']).agg(
    average_value=('Modulus', 'mean'),
    min_value=('Modulus', 'min'),
    max_value=('Modulus', 'max')
).reset_index()

final_df1 = agg1[['ts', 'average_value', 'min_value', 'max_value','label']]
final_df1.to_csv('AGGR_DATA_1.csv', index=False)

df2 = pd.read_csv('ALL_DATA_2.csv')

df2['ts'] = (df2['Time']//0.1)*0.1
df2['ts'] = df2['ts'].round(1)

df2['group'] = (df2['label'] != df2['label'].shift()).cumsum()

agg2 = df2.groupby(['ts', 'group','label']).agg(
    average_value=('Modulus', 'mean'),
    min_value=('Modulus', 'min'),
    max_value=('Modulus', 'max')
).reset_index()

final_df2 = agg2[['ts', 'average_value', 'min_value', 'max_value','label']]
final_df2.to_csv('AGGR_DATA_2.csv', index=False)

"""**merging files**"""

agg_df1 = pd.read_csv('AGGR_DATA_1.csv')
agg_df2 = pd.read_csv('AGGR_DATA_2.csv')

agg_df1 = agg_df1.rename(columns={
    'average_value': 'sensor1_avg_val',
    'min_value': 'sensor1_min_val',
    'max_value': 'sensor1_max_val'
})

agg_df2 = agg_df2.rename(columns={
    'average_value': 'sensor2_avg_val',
    'min_value': 'sensor2_min_val',
    'max_value': 'sensor2_max_val'
})

merged_df = pd.merge(agg_df1, agg_df2, on=['ts', 'label'], how='outer')

merged_df.sort_values(by='ts', inplace=True)

columns = ['ts',
        'sensor1_avg_val', 'sensor1_min_val', 'sensor1_max_val',
        'sensor2_avg_val', 'sensor2_min_val', 'sensor2_max_val',
        'label']

merged_df = merged_df[columns]

change_columns = ['sensor1_avg_val', 'sensor1_min_val', 'sensor1_max_val',
        'sensor2_avg_val', 'sensor2_min_val', 'sensor2_max_val']

merged_df[change_columns] = merged_df[change_columns].astype(object)

merged_df.fillna('?', inplace=True)

merged_df.to_csv('AGGR_DATA_ALL.csv', index=False)

"""**Model creation and training**

**Part 2 A**
"""

finaldf = pd.read_csv('AGGR_DATA_ALL.csv')

finaldf.head()

finaldf.shape

(finaldf == '?').sum()

finaldf = finaldf.replace('?', np.nan)


finaldf.isnull().sum()

cleandf = finaldf.dropna()

cleandf.head()

cleandf.shape


x = cleandf.drop(columns=['ts', 'label'])
y = cleandf['label']

x.head()

y.head()

scaler = StandardScaler()

x_train, x_val_test, y_train, y_val_test = train_test_split(x, y, train_size=0.4, random_state=42, stratify=y)

x_train.shape

x_validation, x_test, y_validation, y_test = train_test_split(x_val_test, y_val_test, test_size=0.5, random_state=42, stratify=y_val_test)

x_test.shape

x_validation.shape

x_train = scaler.fit_transform(x_train)
x_validation = scaler.transform(x_validation)
x_test = scaler.transform(x_test)

nn = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', solver='adam', max_iter=1000, random_state=42)

nn.fit(x_train, y_train)

print({nn.out_activation_})

y_pred = nn.predict(x_test)
y_true = y_test

print("accuracy for base nural network: ", accuracy_score(y_true, y_pred))

print("F1 score for base nural network: ", f1_score(y_true, y_pred, average='weighted'))

"""**Part 2B**

"""

x_test_dt = np.concatenate([x_validation, x_test], axis=0)

y_test_dt = np.concatenate([y_validation, y_test], axis=0)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(x_train, y_train)

y_pred_dt = dt.predict(x_test_dt)

print("accuracy for base decision tree: ", accuracy_score(y_test_dt, y_pred_dt))

print("f1 for base decision tree: ", f1_score(y_test_dt, y_pred_dt, average='weighted'))

"""**Part 3**"""

df1 = pd.read_csv('ALL_DATA_1.csv')
df1['ts'] = (df1['Time']//0.1)*0.1
df1['ts'] = df1['ts'].round(1)
df1['group'] = (df1['label'] != df1['label'].shift()).cumsum()
agg1 = df1.groupby(['ts', 'group', 'label']).agg(
    sensor_1_average_value=('Modulus', 'mean'),
    sensor_1_min_value=('Modulus', 'min'),
    sensor_1_max_value=('Modulus', 'max'),
    sensor_1_std_value=('Modulus', 'std')
).reset_index()

agg1 = agg1.sort_values('ts').reset_index(drop=True)

agg1['sensor_1_max_abs_10s'] = agg1['sensor_1_average_value'].abs().rolling(100, min_periods=1).max()
agg1['sensor_1_min_abs_10s'] = agg1['sensor_1_average_value'].abs().rolling(100, min_periods=1).min()

final_df1 = agg1[['ts', 'sensor_1_average_value', 'sensor_1_min_value', 'sensor_1_max_value', 'sensor_1_std_value', 'sensor_1_max_abs_10s','sensor_1_min_abs_10s', 'label']]
final_df1.to_csv('AGGR_DATA_1_part3.csv', index=False)

df2 = pd.read_csv('ALL_DATA_2.csv')
df2['ts'] = (df2['Time']//0.1)*0.1
df2['ts'] = df2['ts'].round(1)
df2['group'] = (df2['label'] != df2['label'].shift()).cumsum()
agg2 = df2.groupby(['ts', 'group', 'label']).agg(
    sensor_2_average_value=('Modulus', 'mean'),
    sensor_2_min_value=('Modulus', 'min'),
    sensor_2_max_value=('Modulus', 'max'),
    sensor_2_std_value=('Modulus', 'std')
).reset_index()

agg2 = agg2.sort_values('ts').reset_index(drop=True)

agg2['sensor_2_max_abs_10s'] = agg2['sensor_2_average_value'].abs().rolling(100, min_periods=1).max()
agg2['sensor_2_min_abs_10s'] = agg2['sensor_2_average_value'].abs().rolling(100, min_periods=1).min()
final_df2 = agg2[['ts', 'sensor_2_average_value', 'sensor_2_min_value', 'sensor_2_max_value', 'sensor_2_std_value', 'sensor_2_max_abs_10s', 'sensor_2_min_abs_10s','label']]
final_df2.to_csv('AGGR_DATA_2_part3.csv', index=False)

agg_df1 = pd.read_csv('AGGR_DATA_1_part3.csv')
agg_df2 = pd.read_csv('AGGR_DATA_2_part3.csv')

merged_df = pd.merge(agg_df1, agg_df2, on=['ts', 'label'], how='outer')
merged_df.sort_values(by='ts', inplace=True)

merged_df.head()

columns = ['ts','sensor_1_average_value', 'sensor_1_min_value', 'sensor_1_max_value','sensor_1_std_value', 'sensor_1_max_abs_10s', 'sensor_1_min_abs_10s','sensor_2_average_value', 'sensor_2_min_value', 'sensor_2_max_value','sensor_2_std_value', 'sensor_2_max_abs_10s','sensor_2_min_abs_10s', 'label']

merged_df = merged_df[columns]

merged_df.head()

change_columns = ['sensor_1_average_value', 'sensor_1_min_value', 'sensor_1_max_value','sensor_1_std_value', 'sensor_1_max_abs_10s', 'sensor_1_min_abs_10s','sensor_2_average_value', 'sensor_2_min_value', 'sensor_2_max_value','sensor_2_std_value', 'sensor_2_max_abs_10s','sensor_2_min_abs_10s']
merged_df[change_columns] = merged_df[change_columns].astype(object)

merged_df.fillna('?', inplace=True)

merged_df.to_csv('AGGR_DATA_ALL_part3.csv', index=False)

finaldf_part3 = pd.read_csv('AGGR_DATA_ALL_part3.csv')
finaldf_part3.head()

finaldf_part3.shape

(finaldf_part3 == '?').sum()

finaldf_part3 = finaldf_part3.replace('?', np.nan)


finaldf_part3.isnull().sum()

cleandf_part3 = finaldf_part3.dropna()

cleandf_part3.head()

cleandf_part3.shape


x = cleandf_part3.drop(columns=['ts', 'label'])
y = cleandf_part3['label']
x.head()
y.head()
scaler = StandardScaler()
x_train, x_val_test, y_train, y_val_test = train_test_split(x, y, train_size=0.4, random_state=42, stratify=y)

x_train.shape

x_validation, x_test, y_validation, y_test = train_test_split(x_val_test, y_val_test, test_size=0.5, random_state=42,stratify=y_val_test)

x_test.shape

x_validation.shape

x_train = scaler.fit_transform(x_train)
x_validation = scaler.transform(x_validation)
x_test = scaler.transform(x_test)

nn_part3 = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', solver='adam', max_iter=1000, random_state=42)
nn_part3.fit(x_train, y_train)

print({nn_part3.out_activation_})

y_pred = nn_part3.predict(x_test)
y_true = y_test

print("accuracy for hidden_layer_sizes=(10,), activation='relu' with new data features: ", accuracy_score(y_true, y_pred))
print("F1 score for hidden_layer_sizes=(10,), activation='relu' with new data features: ", f1_score(y_true, y_pred, average='weighted'))

x_test_dt_part3 = np.concatenate([x_validation, x_test], axis=0)
y_test_dt_part3 = np.concatenate([y_validation, y_test], axis=0)
dt = DecisionTreeClassifier(random_state=42)
dt.fit(x_train, y_train)
y_pred_dt = dt.predict(x_test_dt_part3)
print("accuracy for decision tree with new data features: ", accuracy_score(y_test_dt_part3, y_pred_dt))
print("f1 for decision tree with new data features: ", f1_score(y_test_dt_part3, y_pred_dt, average='weighted'))

"""**Part 3B1 (different topologies)**"""

nn_part3b = MLPClassifier(hidden_layer_sizes=(64,32), activation='relu', solver='adam', max_iter=1000, random_state=42)
nn_part3b.fit(x_train, y_train)

y_pred = nn_part3b.predict(x_test)
y_true = y_test

print("accuracy for hidden_layer_sizes=(64,32), activation='relu': ", accuracy_score(y_true, y_pred))
print("F1 score for hidden_layer_sizes=(64,32), activation='relu': ", f1_score(y_true, y_pred, average='weighted'))

nn_part3b = MLPClassifier(hidden_layer_sizes=(64, 32), activation='tanh', solver='adam', max_iter=1000, random_state=42)
nn_part3b.fit(x_train, y_train)
y_pred = nn_part3b.predict(x_test)
y_true = y_test
print("accuracy for hidden_layer_sizes=(64, 32), activation='tanh': ", accuracy_score(y_true, y_pred))
print("F1 score for hidden_layer_sizes=(64, 32), activation='tanh': ", f1_score(y_true, y_pred, average='weighted'))

nn_part3b = MLPClassifier(hidden_layer_sizes=(256, 100, 50), activation='tanh', solver='adam', max_iter=1000, random_state=42)
nn_part3b.fit(x_train, y_train)
y_pred = nn_part3b.predict(x_test)
y_true = y_test
print("accuracy for hidden_layer_sizes=(256, 100, 50), activation='tanh': ", accuracy_score(y_true, y_pred))
print("F1 score for hidden_layer_sizes=(256, 100, 50), activation='tanh': ", f1_score(y_true, y_pred, average='weighted'))

"""**3B2 (CNN)**"""

import os
os.environ["KERAS_BACKEND"] = "torch"
import keras

model = keras.Sequential([
    keras.layers.Input((12, 1)),
    keras.layers.Conv1D(32, 3, activation='relu', padding='same'),
    keras.layers.Conv1D(64, 3, activation='relu', padding='same'),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(5, activation='softmax'),
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
res = model.evaluate(x_test, y_test)
print('test loss, acc=', res)
y_pred_cnn = model.predict(x_test).argmax(axis=1)
print("accuracy for CNN: ", accuracy_score(y_test, y_pred_cnn))
print("F1 score for CNN: ", f1_score(y_test, y_pred_cnn, average='weighted'))

