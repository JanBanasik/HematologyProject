#!/usr/bin/env python3
"""
Script to extract CBC parameters and anemia labels from MIMIC-IV Clinical Database Demo CSVs.

Required input files (place in 'hosp/' subdirectory alongside this script):
  - hosp/d_labitems.csv         # definitions of lab tests
  - hosp/labevents.csv         # lab results
  - hosp/diagnoses_icd.csv     # ICD diagnoses

Output:
  - cbc_anemia_dataset.csv     # one row per patient with CBC means and multi-class label

Usage:
  python extract_cbc_anemia.py
"""
import os
import pandas as pd
import numpy as np

# --- File paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HOSP_DIR = os.path.join(SCRIPT_DIR, 'hosp')
LABITEMS_FP = os.path.join(HOSP_DIR, 'd_labitems.csv')
LABEVENTS_FP = os.path.join(HOSP_DIR, 'labevents.csv')
DIAGNOSES_FP = os.path.join(HOSP_DIR, 'diagnoses_icd.csv')
OUTPUT_FP = os.path.join(SCRIPT_DIR, 'cbc_anemia_dataset.csv')

# --- Define CBC parameters and search keys ---
TARGETS = {
    'HGB': 'Hemoglobin',
    'RBC': 'Red Blood Cells',
    'HCT': 'Hematocrit',
    'MCV': 'MCV',
    'MCH': 'MCH',
    'MCHC': 'MCHC',
    'RDW': 'RDW',
    'WBC': 'White Blood Cells',
    'PLT': 'Platelet'
}

# --- Load lab test definitions and identify ITEMIDs ---
def get_cbc_itemids():
    df = pd.read_csv(LABITEMS_FP)
    # detect label column
    name_col = next((col for col in df.columns if col.lower()=='label' or col.lower()=='label'), None)
    if name_col is None:
        raise KeyError(f"Nie znaleziono kolumny 'LABEL' w {LABITEMS_FP}. Dostępne kolumny: {df.columns.tolist()}")
    # detect itemid column
    itemid_col = next((col for col in df.columns if col.lower()=='itemid'), None)
    if itemid_col is None:
        raise KeyError(f"Nie znaleziono kolumny 'ITEMID' w {LABITEMS_FP}. Dostępne kolumny: {df.columns.tolist()}")
    itemids = {}
    for param, key in TARGETS.items():
        hits = df[df[name_col].str.contains(key, case=False, na=False)]
        itemids[param] = hits[itemid_col].unique().tolist()
        print(f"{param}: found ITEMIDs {itemids[param]}")
    return itemids

# --- Extract lab results for those ITEMIDs ---
def extract_cbc_results(itemids):
    # Detect correct column names in labevents.csv
    sample = pd.read_csv(LABEVENTS_FP, nrows=0)
    cols0 = sample.columns.tolist()
    subj_col = next((c for c in cols0 if c.lower()=='subject_id'), None)
    item_col = next((c for c in cols0 if c.lower()=='itemid'), None)
    # Only use numeric VALUENUM
    val_col = next((c for c in cols0 if c.lower()=='valuenum'), None)
    if not subj_col or not item_col or not val_col:
        raise KeyError(f"Nie znaleziono wymaganych kolumn w labevents.csv: {cols0}")
    # Read in chunks using detected column names
    parts = []
    flat_ids = [iid for ids in itemids.values() for iid in ids]
    for chunk in pd.read_csv(LABEVENTS_FP, usecols=[subj_col, item_col, val_col], chunksize=500_000):
        # Rename to standard
        chunk = chunk.rename(columns={subj_col:'SUBJECT_ID', item_col:'ITEMID', val_col:'VALUENUM'})
        # Ensure numeric, coerce errors
        chunk['VALUENUM'] = pd.to_numeric(chunk['VALUENUM'], errors='coerce')
        chunk = chunk.dropna(subset=['VALUENUM'])
        mask = chunk['ITEMID'].isin(flat_ids)
        parts.append(chunk.loc[mask, ['SUBJECT_ID','ITEMID','VALUENUM']])
    lab = pd.concat(parts, ignore_index=True)
    return lab

# --- Map ITEMID back to parameter name and aggregate ---
def build_cbc_table(lab, itemids):
    # reverse mapping
    reverse = {iid: param for param, ids in itemids.items() for iid in ids}
    lab['param'] = lab['ITEMID'].map(reverse)
    # drop nulls
    lab = lab.dropna(subset=['param'])
    # average per patient and parameter
    cbc = lab.groupby(['SUBJECT_ID','param'])['VALUENUM'] \
             .mean().unstack()
    cbc.reset_index(inplace=True)
    return cbc

# --- Build anemia labels from diagnoses ---
def build_labels():
    # Load diagnoses, detect columns
    sample = pd.read_csv(DIAGNOSES_FP, nrows=0)
    cols0 = sample.columns.tolist()
    subj_col = next((c for c in cols0 if c.lower()=='subject_id'), None)
    icd_col = next((c for c in cols0 if 'icd' in c.lower()), None)
    if not subj_col or not icd_col:
        raise KeyError(f"Nie znaleziono wymaganych kolumn w {DIAGNOSES_FP}: {cols0}")
    dx = pd.read_csv(DIAGNOSES_FP, usecols=[subj_col, icd_col])
    dx = dx.rename(columns={subj_col:'SUBJECT_ID', icd_col:'ICD_CODE'})
    # define code groups
    groups = [
        ('Anemia Mikrocytarna', lambda code: str(code).startswith('D50')),
        ('Anemia Makrocytarna', lambda code: str(code).startswith('D51')),
        ('Anemia Hemolityczna', lambda code: str(code).startswith(('D55','D56','D57','D58','D59'))),
        ('Anemia Aplastyczna', lambda code: str(code).startswith('D61')),
        ('Anemia Normocytarna', lambda code: str(code).startswith('D63'))
    ]
    # assign type per row
    def map_type(icd):
        for label, fn in groups:
            if fn(icd): return label
        return None
    dx['anemia_type'] = dx['ICD_CODE'].apply(map_type)
    dx_filtered = dx.dropna(subset=['anemia_type']).copy()
    # priority by group order
    pri = {label: idx for idx, (label, _) in enumerate(groups)}
    dx_filtered.loc[:, 'priority'] = dx_filtered['anemia_type'].map(pri)
    idx = dx_filtered.groupby('SUBJECT_ID')['priority'].idxmin()
    dx_min = dx_filtered.loc[idx]
    labels = dx_min.set_index('SUBJECT_ID')['anemia_type']
    return labels

# --- Main ---
if __name__ == '__main__':
    print('Identifying CBC ITEMIDs...')
    itemids = get_cbc_itemids()
    print('Extracting lab results...')
    lab = extract_cbc_results(itemids)
    print('Building CBC table...')
    cbc = build_cbc_table(lab, itemids)
    print(f'CBC table shape: {cbc.shape}')
    print('Building anemia labels...')
    labels = build_labels()
    # merge
    df = cbc.merge(labels.rename('label'), left_on='SUBJECT_ID', right_index=True, how='left')
    # fill healthy
    df['label'] = df['label'].fillna('Healthy')

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_save_file_name = os.path.join(BASE_DIR, "data", "anemia", "dane-anemia-extracted.csv")

    df.to_csv(data_save_file_name, index=False)
    print(f'Final dataset saved to {data_save_file_name}')
