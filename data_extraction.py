import os
import torch
import random
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# eICU
def dataframe_from_csv(path, header=0, index_col=False):
    return pd.read_csv(path, header=header, index_col=index_col)


def read_diagnosis_table(eicu_path):
    diag = dataframe_from_csv(os.path.join(eicu_path, 'diagnosis.csv'), index_col=False)
    diag = diag[diag["diagnosisoffset"] > 0]
    diag = diag[['patientunitstayid', 'activeupondischarge', 'diagnosisoffset',
                'diagnosisstring', 'icd9code']]
    diag = diag[diag['icd9code'].notnull()]
    tes = diag['icd9code'].str.split(pat=",", expand=True, n=1)
    labels_name = ["Shock","Septicemia","Respiratory failure","Pneumonia","Pleurisy",
              "upper respiratory","lower respiratory","Other liver diseases",
              "Hypertension with complications","Gastrointestinal hem",
              "Fluid disorders","Essential hypertension","lipid disorder",
              "DM without complication","DM with complications",
              "Coronary athe","CHF", "Conduction disorders","Complications of surgical",
              "COPD", "CKD", "Cardiac dysrhythmias","Acute myocardial infarction",
               "Acute cerebrovascular disease","Acute and unspecified renal failure"]
    diag['icd0'] = np.nan
    diag['icd0'] = tes[0]
    diag['icd'] = np.nan
    diag['icd'] = diag['icd0'].str.replace('.', '')
    diag = diag.reindex(columns=diag.columns.tolist() + labels_name)
    diag[labels_name] = np.nan
    return diag


def diag_labels(diag):
    import json
    codes = json.load(open('phen_code.json'))
    diag.loc[diag['icd'].isin(codes['septicemia']), 'Septicemia'] = 1
    diag.loc[diag['icd'].isin(codes['Shock']), 'Shock'] = 1
    diag.loc[diag['icd'].isin(codes['Compl_surgical']), 'Complications of surgical'] = 1
    diag.loc[diag['icd'].isin(codes['ckd']), 'CKD'] = 1
    diag.loc[diag['icd'].isin(codes['renal_failure']), 'Acute and unspecified renal failure'] = 1

    diag.loc[diag['icd'].isin(codes['Gastroint_hemorrhage']), 'Gastrointestinal hem'] = 1
    diag.loc[diag['icd'].isin(codes['Other_liver_dis']), 'Other liver diseases'] = 1
    diag.loc[diag['icd'].isin(codes['upper_respiratory']), 'upper respiratory'] = 1
    diag.loc[diag['icd'].isin(codes['lower_respiratory']), 'lower respiratory'] = 1
    diag.loc[diag['icd'].isin(codes['Resp_failure']), 'Respiratory failure'] = 1

    diag.loc[diag['icd'].isin(codes['Pleurisy']), 'Pleurisy'] = 1
    diag.loc[diag['icd'].isin(codes['COPD']), 'COPD'] = 1
    diag.loc[diag['icd'].isin(codes['Pneumonia']), 'Pneumonia'] = 1
    diag.loc[diag['icd'].isin(codes['Acute_cerebrovascular']), 'Acute cerebrovascular disease'] = 1
    diag.loc[diag['icd'].isin(codes['Congestive_hf']), 'CHF'] = 1

    diag.loc[diag['icd'].isin(codes['Cardiac_dysr']), 'Cardiac dysrhythmias'] = 1
    diag.loc[diag['icd'].isin(codes['Conduction_dis']), 'Conduction disorders'] = 1
    diag.loc[diag['icd'].isin(codes['Coronary_ath']), 'Coronary athe'] = 1
    diag.loc[diag['icd'].isin(codes['myocar_infarction']), 'Acute myocardial infarction'] = 1
    diag.loc[diag['icd'].isin(codes['hypercomp']), 'Hypertension with complications'] = 1

    diag.loc[diag['icd'].isin(codes['essehyper']), 'Essential hypertension'] = 1
    diag.loc[diag['icd'].isin(codes['fluiddiso']), 'Fluid disorders'] = 1
    diag.loc[diag['icd'].isin(codes['lipidmetab']), 'lipid disorder'] = 1
    diag.loc[diag['icd'].isin(codes['t2dmcomp']), 'DM with complications'] = 1
    diag.loc[diag['icd'].isin(codes['t2dmwocomp']), 'DM without complication'] = 1
    return diag


def embedding(root_dir):
    all_df = prepare_categorical_variables(root_dir)
    return all_df


def prepare_categorical_variables(root_dir):
    columns_ord = [ 'patientunitstayid', 'itemoffset',
    'Eyes', 'Motor', 'GCS Total', 'Verbal',
    'ethnicity', 'gender','apacheadmissiondx',
    'FiO2','Heart Rate', 'Invasive BP Diastolic',
    'Invasive BP Systolic', 'MAP (mmHg)',  'O2 Saturation',
    'Respiratory Rate', 'Temperature (C)', 'admissionheight',
    'admissionweight', 'age', 'glucose', 'pH',
    'hospitaladmitoffset',
    'hospitaldischargestatus','unitdischargeoffset',
    'unitdischargestatus']
    all_df = pd.read_csv(os.path.join(root_dir, 'all_data.csv'))

    all_df = all_df[all_df.gender != 0]
    all_df = all_df[all_df.hospitaldischargestatus != 2]
    all_df = all_df[columns_ord]

    all_df.apacheadmissiondx = all_df.apacheadmissiondx.astype(int)
    all_df.ethnicity = all_df.ethnicity.astype(int)
    all_df.gender = all_df.gender.astype(int)
    all_df['GCS Total'] = all_df['GCS Total'].astype(int)
    all_df['Eyes'] = all_df['Eyes'].astype(int)
    all_df['Motor'] = all_df['Motor'].astype(int)
    all_df['Verbal'] = all_df['Verbal'].astype(int)

    return all_df


def filter_mortality_data(all_df):
    all_df = all_df[all_df.gender != 0]
    all_df = all_df[all_df.hospitaldischargestatus!=2]
    all_df['unitdischargeoffset'] = all_df['unitdischargeoffset']/(1440)
    all_df['itemoffsetday'] = (all_df['itemoffset']/24)
    all_df.drop(columns='itemoffsetday',inplace=True)
    mort_cols = ['patientunitstayid', 'itemoffset', 'apacheadmissiondx', 'ethnicity','gender',
                'GCS Total', 'Eyes', 'Motor', 'Verbal',
                'admissionheight','admissionweight', 'age', 'Heart Rate', 'MAP (mmHg)',
                'Invasive BP Diastolic', 'Invasive BP Systolic', 'O2 Saturation',
                'Respiratory Rate', 'Temperature (C)', 'glucose', 'FiO2', 'pH',
                'unitdischargeoffset','hospitaldischargestatus']

    all_mort = all_df[mort_cols]
    all_mort = all_mort[all_mort['unitdischargeoffset'] >=2]
    all_mort = all_mort[all_mort['itemoffset']> 0]
    return all_mort

def data_extraction_mortality(time_window,root_dir):
    all_df = embedding(root_dir)
    all_mort = filter_mortality_data(all_df)
    all_mort = all_mort[all_mort['itemoffset']<=time_window]
    return all_mort


def data_extraction_ICD(all_df, eicu_dir):
    label_pheno = ['Respiratory failure', 'Essential hypertension',
                   'Cardiac dysrhythmias', 'Fluid disorders', 'Septicemia',
                   'Acute and unspecified renal failure', 'Pneumonia',
                   'Acute cerebrovascular disease', 'CHF', 'CKD', 'COPD',
                   'Acute myocardial infarction', "Gastrointestinal hem",
                   'Shock', 'lipid disorder', 'DM with complications', 'Coronary athe',
                   'Pleurisy', 'Other liver diseases', 'lower respiratory',
                   'Hypertension with complications', 'Conduction disorders',
                   'Complications of surgical', 'upper respiratory',
                   'DM without complication']

    diag_ord_col = ["patientunitstayid", "itemoffset", "Respiratory failure", "Fluid disorders",
                    "Septicemia", "Acute and unspecified renal failure", "Pneumonia",
                    "Acute cerebrovascular disease",
                    "Acute myocardial infarction", "Gastrointestinal hem", "Shock", "Pleurisy",
                    "lower respiratory", "Complications of surgical", "upper respiratory",
                    "Hypertension with complications", "Essential hypertension", "CKD", "COPD",
                    "lipid disorder", "Coronary athe", "DM without complication",
                    "Cardiac dysrhythmias",
                    "CHF", "DM with complications", "Other liver diseases", "Conduction disorders"]

    diag_columns = ['patientunitstayid', 'itemoffset', 'Respiratory failure', 'Essential hypertension',
                    'Cardiac dysrhythmias',
                    'Fluid disorders', 'Septicemia', 'Acute and unspecified renal failure', 'Pneumonia',
                    'Acute cerebrovascular disease', 'CHF', 'CKD', 'COPD', 'Acute myocardial infarction',
                    "Gastrointestinal hem",
                    'Shock', 'lipid disorder', 'DM with complications', 'Coronary athe', 'Pleurisy',
                    'Other liver diseases', 'lower respiratory',
                    'Hypertension with complications', 'Conduction disorders', 'Complications of surgical',
                    'upper respiratory',
                    'DM without complication']

    codes = json.load(open('phen_code.json'))

    diag = read_diagnosis_table(eicu_dir)
    diag = diag_labels(diag)
    diag.dropna(how='all', subset=label_pheno, inplace=True)

    stay_diag = set(diag['patientunitstayid'].unique())
    stay_all = set(all_df.patientunitstayid.unique())
    stay_intersection = stay_all.intersection(stay_diag)
    stay_pheno = list(stay_intersection)

    diag = diag[diag['patientunitstayid'].isin(stay_pheno)]
    diag.rename(index=str, columns={"diagnosisoffset": "itemoffset"}, inplace=True)
    diag = diag[diag_columns]
    label = diag.groupby('patientunitstayid').sum()
    label = label.reset_index()
    label[label_pheno] = np.where(label[label_pheno] >= 1, 1, label[label_pheno])
    all_pheno = all_df[all_df["patientunitstayid"].isin(stay_pheno)]
    all_pheno = all_pheno[all_pheno["itemoffset"] > 0]
    label = label[diag_ord_col]
    all_pheno_label = label[label.patientunitstayid.isin(list(all_pheno.patientunitstayid.unique()))]

    df = all_pheno.merge(all_pheno_label.drop(columns=['itemoffset']), on='patientunitstayid')

    return df


def df_to_list(df):
    grp_df = df.groupby('patientunitstayid')
    df_arr = []
    for idx, frame in grp_df:
        df_arr.append(frame)

    return df_arr


def pad(data, max_len=200):
    padded_data = []
    nrows = []
    for item in data:
        if item.shape[0] > 200:
            continue

        tmp = np.zeros((max_len, item.shape[1]))
        tmp[:item.shape[0], :item.shape[1]] = item
        padded_data.append(tmp)
        nrows.append(item.shape[0])
    padded_data = np.array(padded_data)

    return padded_data, nrows


def normalize_data_mort(data, train_idx, valid_idx, test_idx):
    train = data[data['patientunitstayid'].isin(train_idx)]
    valid = data[data['patientunitstayid'].isin(valid_idx)]
    test = data[data['patientunitstayid'].isin(test_idx)]

    col_used = ['patientunitstayid', 'itemoffset']

    dec_cat = ['GCS Total', 'Eyes', 'Motor', 'Verbal']
    dec_num = ['admissionheight', 'admissionweight', 'Heart Rate', 'MAP (mmHg)', 'Invasive BP Diastolic',
               'Invasive BP Systolic',
               'O2 Saturation', 'Respiratory Rate', 'Temperature (C)', 'glucose', 'FiO2', 'pH']
    diag_col = ['Respiratory failure', 'Essential hypertension', 'Cardiac dysrhythmias',
                'Fluid disorders', 'Septicemia', 'Acute and unspecified renal failure', 'Pneumonia',
                'Acute cerebrovascular disease', 'CHF', 'CKD', 'COPD', 'Acute myocardial infarction',
                "Gastrointestinal hem",
                'Shock', 'lipid disorder', 'DM with complications', 'Coronary athe', 'Pleurisy', 'Other liver diseases',
                'lower respiratory',
                'Hypertension with complications', 'Conduction disorders', 'Complications of surgical',
                'upper respiratory',
                'DM without complication']

    demo_col = ['age', 'gender', 'ethnicity']
    col_used += dec_cat
    col_used += dec_num
    col_used += diag_col
    col_used += demo_col
    col_used += ['hospitaldischargestatus']

    train = train[col_used]
    valid = valid[col_used]
    test = test[col_used]

    train = df_to_list(train)
    valid = df_to_list(valid)
    test = df_to_list(test)

    train, nrows_train = pad(train)
    valid, nrows_valid = pad(valid)
    test, nrows_test = pad(test)

    return (train, nrows_train), (valid, nrows_valid), (test, nrows_test)


def get_task_specific_labels(data):
    return data[:, 0, :].astype(int)


def get_data_generator(data, label, static, nrows, batch_size, train=True):
    data_gen = batch_generator(data, label, static, nrows=nrows, batch_size=batch_size)
    steps = np.ceil(len(data) / batch_size)
    return data_gen, int(steps)


def batch_generator(data, labels, static, nrows=None, batch_size=256, rng=np.random.RandomState(0), shuffle=True,
                    sample=False):
    while True:
        if shuffle:
            d = list(zip(data, labels, static, nrows))
            random.shuffle(d)
            data, labels, static, nrows = zip(*d)
        data = np.stack(data)
        labels = np.stack(labels)
        static = np.stack(static)
        for i in range(0, len(data), batch_size):
            x_batch = data[i:i + batch_size]
            y_batch = labels[i:i + batch_size]
            static_batch = static[i:i + batch_size]
            if nrows:
                nrows_batch = np.array(nrows)[i:i + batch_size]

            x_cat = x_batch[:, :, 1:5].astype(int)
            x_num = x_batch[:, :, 5:]
            x_time = x_batch[:, :, 0]
            yield [x_cat, x_num], y_batch, static_batch, nrows_batch, x_time


def get_data(train, valid, test, batch_size):
    nrows_train = train[1]
    nrows_valid = valid[1]
    nrows_test = test[1]
    n_labels = 1

    X_train = train[0][:, :, 1:18]
    X_valid = valid[0][:, :, 1:18]
    X_test = test[0][:, :, 1:18]
    static_train = train[0][:, :, 18:-1]
    static_valid = valid[0][:, :, 18:-1]
    static_test = test[0][:, :, 18:-1]

    Y_train = get_task_specific_labels(train[0][:, :, -n_labels:])
    Y_valid = get_task_specific_labels(valid[0][:, :, -n_labels:])
    Y_test = get_task_specific_labels(test[0][:, :, -n_labels:])

    train_gen, train_steps = get_data_generator(X_train, Y_train, static_train, nrows_train, batch_size)
    valid_gen, valid_steps = get_data_generator(X_valid, Y_valid, static_valid, nrows_valid, batch_size)
    test_gen, test_steps = get_data_generator(X_test, Y_test, static_test, nrows_test, batch_size)

    return train_gen, train_steps, valid_gen, valid_steps, test_gen, test_steps


def data_process_eICU(data_path, data_s_path, time_length):
    batch_size = 256
    df_data = data_extraction_mortality(time_length, data_path)
    df_data = df_data.reset_index(drop=True)
    data = data_extraction_ICD(df_data, data_s_path)
    all_idx = np.array(list(data['patientunitstayid'].unique()))
    data = data.reset_index(drop=True)

    train_idx, valid_test_idx = train_test_split(all_idx, test_size=0.3, random_state=1)
    valid_idx, test_idx = train_test_split(valid_test_idx, test_size=0.5, random_state=1)

    train, valid, test = normalize_data_mort(data, train_idx, valid_idx, test_idx)
    train_gen, train_steps, valid_gen, valid_steps, test_gen, test_steps = get_data(train, valid, test, batch_size)

    return train_gen, train_steps, valid_gen, valid_steps, test_gen, test_steps

