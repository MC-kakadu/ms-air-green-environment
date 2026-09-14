import pandas as pd

# Raw data convert
# Convert the file format from SAS to CSV
def sas_to_csv(sas_filepath, csv_filepath):
    # SAS file loading (for the euc-kr encoding)
    df_sas = pd.read_sas(sas_filepath, format='sas7bdat', encoding='euc-kr')
    # Convert and save to CSV file
    df_sas.to_csv(csv_filepath, index=False, encoding='utf-8-sig')
    print(f"SAS converting Completed!: {csv_filepath}") 

sas_to_csv('/content/drive/MyDrive/MetS_Experiment/data/hn18_all.sas7bdat', '/content/drive/MyDrive/MetS_Experiment/data/hn_18_all.csv')   # target file and path of the year 2018
sas_to_csv('/content/drive/MyDrive/MetS_Experiment/data/hn19_all.sas7bdat', '/content/drive/MyDrive/MetS_Experiment/data/hn_19_all.csv')   # target file and path of the year 2019
sas_to_csv('/content/drive/MyDrive/MetS_Experiment/data/hn22_all.sas7bdat', '/content/drive/MyDrive/MetS_Experiment/data/hn_22_all.csv')   # target file and path of the year 2022

# Convert the file format from SPSS to CSV
def spss_to_csv(sav_filepath, csv_filepath):
    df_spss = pd.read_spss(sav_filepath)
    df_spss.to_csv(csv_filepath, index=False, encoding='utf-8-sig')
    print(f"SPSS converting Completed!: {csv_filepath}")

spss_to_csv('/content/drive/MyDrive/MetS_Experiment/data/HN20_all.sav', '/content/drive/MyDrive/MetS_Experiment/data/hn_20_all.csv')   # target filr and path of the year 2020
spss_to_csv('/content/drive/MyDrive/MetS_Experiment/data/HN21_all.sav', '/content/drive/MyDrive/MetS_Experiment/data/hn_21_all.csv')   # target filr and path of the year 2021

columns_to_use = [
    'year', 'sex', 'age', 'edu', 'marri_1', 'ho_incm', 'sm_presnt', 
    'dr_month', 'BD1_11', 'tins', 'npins', 'EC1_1', 'BE3_31', 
    'pa_aerobic', 'mh_stress', 
    'wt_itvex', 'kstrata', 'psu', 'HE_obe', 'HE_wc', 'HE_TG', 'HE_HDL_st2', 'HE_sbp', 'HE_dbp', 'HE_glu', 'region'  
]

df_2018 = pd.read_csv('/content/drive/MyDrive/MetS_Experiment/data/hn_18_all.csv', usecols=columns_to_use)
df_2019 = pd.read_csv('/content/drive/MyDrive/MetS_Experiment/data/hn_19_all.csv', usecols=columns_to_use)
df_2020 = pd.read_csv('/content/drive/MyDrive/MetS_Experiment/data/hn_20_all.csv', usecols=columns_to_use)
df_2021 = pd.read_csv('/content/drive/MyDrive/MetS_Experiment/data/hn_21_all.csv', usecols=columns_to_use)
df_2022 = pd.read_csv('/content/drive/MyDrive/MetS_Experiment/data/hn_22_all.csv', usecols=columns_to_use)

# Concat all of 5 years data
df_integrated = pd.concat([df_2018, df_2019, df_2020, df_2021, df_2022], ignore_index=True)

# Calculate five-year integrated weight (wt_pool)
df_integrated['wt_pool'] = df_integrated['wt_itvex'] / 5

print("Concat data shape:", df_integrated.shape)
print(df_integrated[['year', 'wt_itvex', 'wt_pool', 'kstrata', 'psu']].head())

df_integrated.to_csv('knhanes_integrated_2018_2022.csv', index=False, encoding='utf-8-sig')
print(" 'knhanes_integrated_2018_2022.csv' file saved.")


# Merge an air pollution data
df_air = pd.read_csv('/content/drive/MyDrive/MetS_Experiment/data/Air_pollution_concentration_by_year.csv')   # air pollution data path

df_air_melted = df_air.melt(id_vars=['region', 'item'], 
                            value_vars=['2018', '2019', '2020', '2021', '2022'],
                            var_name='year', 
                            value_name='value')

df_air_melted['year'] = df_air_melted['year'].astype(float)

df_air_pivoted = df_air_melted.pivot_table(index=['region', 'year'], 
                                           columns='item', 
                                           values='value').reset_index()

df_air_pivoted.columns.name = None

df_knhanes = pd.read_csv('knhanes_integrated_2018_2022.csv')

df_merged = pd.merge(df_knhanes, df_air_pivoted, on=['region', 'year'], how='left')

print("Merged data shape:", df_merged.shape)
print(df_merged[['year', 'region', 'SO2', 'NO2', 'O3', 'CO', 'PM10', 'PM2_5']].head())

df_merged.to_csv('knhanes_air_merged_2018_2022_01.csv', index=False, encoding='utf-8-sig')
print(" 'knhanes_air_merged_2018_2022_01.csv' file saved.")


# Merge the green area data
df_merged = pd.read_csv('/content/drive/MyDrive/MetS_Experiment/data/knhanes_air_merged_2018_2022_01.csv')

df_green = pd.read_csv('/content/drive/MyDrive/MetS_Experiment/data/green_area(2018_2023).csv')

df_green_melted = df_green.melt(
    id_vars=['region'], 
    value_vars=['2018', '2019', '2020', '2021', '2022'],
    var_name='year', 
    value_name='green_1'
)

df_green_melted['year'] = df_green_melted['year'].astype(float)

df_final = pd.merge(df_merged, df_green_melted, on=['region', 'year'], how='left')

print("Merged data shape:", df_final.shape)
print(df_final[['year', 'region', 'SO2', 'PM10', 'green_1']].head())

df_final.to_csv('knhanes_final_dataset_2018_2022.csv', index=False, encoding='utf-8-sig')
print(" 'knhanes_final_dataset_2018_2022.csv' file save.")


# Missing value removal
df_final = pd.read_csv('/content/drive/MyDrive/MetS_Experiment/data/knhanes_final_dataset_2018_2022.csv')

initial_size = len(df_final)
print(f"Total number of subjects before removal of missing values: {initial_size} people")

missing_counts = df_final.isnull().sum()
missing_percentages = (missing_counts / initial_size) * 100

missing_summary = pd.DataFrame({
    'Missing_Count': missing_counts,
    'Missing_Percentage(%)': missing_percentages
})
print("\n[Missing values Summary by variables]")
print(missing_summary[missing_summary['Missing_Count'] > 0])

df_dropped = df_final.dropna()

final_size = len(df_dropped)
print(f"\nThe final data number after the deletion of missing values: {final_size} people")
print(f"Excluded data number: {initial_size - final_size} people")

df_dropped.to_csv('knhanes_dropna_2018_2022.csv', index=False, encoding='utf-8-sig')
print("\nMissing values treatment completed and 'knhanes_dropna_2018_2022.csv' file saved.")


# Excluding 'Unknown' and 'No response' data of 'BE3_31' variable
df = pd.read_csv('/content/drive/MyDrive/MetS_Experiment/data/knhanes_dropna_2018_2022.csv')

initial_size = len(df)
print(f"Data number before the treatment: {initial_size} people")

target_count = len(df[df['BE3_31'] == 99.0])
print(f"Excluding target nnumber: {target_count} people")

df_filtered = df[df['BE3_31'] != 99.0]

final_size = len(df_filtered)
print(f"\nData number after the treatment: {final_size} people")

df_filtered.to_csv('knhanes_final_ready.csv', index=False, encoding='utf-8-sig')
print("\nThe treatment completed and 'knhanes_final_ready.csv' file saved.")


# Filtering only the adult same or above 30 ages (age >= 30)
df = pd.read_csv('/content/drive/MyDrive/MetS_Experiment/data/knhanes_final_ready.csv')

initial_size = len(df)
print(f"Data number before the filtering: {initial_size} people")

df_over_30 = df[df['age'] >= 30.0]

final_size = len(df_over_30)
excluded_count = initial_size - final_size
print(f"Excluded data number: {excluded_count} people")
print(f"Data number after the filtering (30 >= age): {final_size} people")

df_over_30.to_csv('knhanes_over30_ready.csv', index=False, encoding='utf-8-sig')
print("\nThe treatment completed and 'knhanes_over30_ready.csv' file saved.")



# Excluding 'Unknown' and 'No response' data of 'marri_1, BD1_11, tins, npins, EC1_1' variables
df = pd.read_csv('/content/drive/MyDrive/MetS_Experiment/data/knhanes_over30_ready.csv')

initial_size = len(df)
print(f"Data number before the treatment: {initial_size} people")

df_cleaned = df[
    (df['marri_1'] != 9) & (df['marri_1'] != 9.0) &
    (~df['BD1_11'].isin([8, 9, 8.0, 9.0])) &
    (df['tins'] != 99) & (df['tins'] != 99.0) &
    (df['npins'] != 9) & (df['npins'] != 9.0) &
    (df['EC1_1'] != 9) & (df['EC1_1'] != 9.0)
]

final_size = len(df_cleaned)
excluded_count = initial_size - final_size
print(f"Excluded data number: {excluded_count} people")
print(f"Data number after the treatment: {final_size} people")

df_cleaned.to_csv('knhanes_final_cleaned.csv', index=False, encoding='utf-8-sig')
print("\nThe final treatment completed and 'knhanes_final_cleaned.csv' file saved.")


# One-Hot Encoding
df = pd.read_csv('/content/drive/MyDrive/MetS_Experiment/data/knhanes_final_cleaned.csv')

categorical_cols = [
    'sex', 'edu', 'marri_1', 'ho_incm', 'sm_presnt', 
    'dr_month', 'BD1_11', 'tins', 'npins', 'EC1_1', 
    'BE3_31', 'pa_aerobic', 'mh_stress'
]

print(f"Variables number before the encoding: {df.shape[1]} variables")

# drop_first=True: Excluding the first category to prevent multicollinearity (prevent dummy variable trap)
# dtype=int: Return value in integer form of 1/0 instead of True/False
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

print(f"Variables number after the encoding: {df_encoded.shape[1]} variables")

tins_cols = [col for col in df_encoded.columns if 'tins' in col]
print(f"\n[Encoding results of the 'tins' variable]\n generated column: {tins_cols}")
print(df_encoded[tins_cols].head())

df_encoded.to_csv('knhanes_encoded_final.csv', index=False, encoding='utf-8-sig')
print("\nThe treatment is completed and 'knhanes_encoded_final.csv' file saved.")