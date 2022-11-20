import pandas as pd
import numpy as np
import xgboost
from sdv import SDV, Metadata
from sdv.evaluation import evaluate
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sdv.constraints import FixedIncrements
from sdv.constraints import Inequality
from sdv.constraints import FixedCombinations
from sdv.tabular import CopulaGAN
from sdv.tabular import CTGAN
from sdv.tabular import GaussianCopula
from sdv.tabular import TVAE
from sdv.lite import TabularPreset
from dython import nominal
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(5)

df1_300 = pd.read_csv("xgboost/Data/Dataset1/train_data.csv")
df2_70 = pd.read_csv("xgboost/Data/Dataset2/cardio_train.csv")
df3_5 = pd.read_csv("xgboost/Data/Dataset3/healthcare-dataset-stroke-data.csv")

df1_300['Hospital_code'] = df1_300['Hospital_code'].astype(str)
df1_300['Hospital_type_code'] = df1_300['Hospital_type_code'].astype(str)
df1_300['City_Code_Hospital'] = df1_300['City_Code_Hospital'].astype(str)
df1_300['Hospital_region_code'] = df1_300['Hospital_region_code'].astype(str)
df1_300['Department'] = df1_300['Department'].astype(str)
df1_300['Ward_Type'] = df1_300['Ward_Type'].astype(str)
df1_300['Ward_Facility_Code'] = df1_300['Ward_Facility_Code'].astype(str)
df1_300['Bed Grade'] = df1_300['Bed Grade'].astype(str)
df1_300['City_Code_Patient'] = df1_300['City_Code_Patient'].astype(str)
df1_300['Type of Admission'] = df1_300['Type of Admission'].astype(str)
df1_300['Severity of Illness'] = df1_300['Severity of Illness'].astype(str)
df1_300['Age'] = df1_300['Age'].astype(str)

df1_los_rounding = FixedIncrements(column_name="LOS", increment_value=10)
df2_ap_inequality = Inequality(high_column_name="ap_hi", low_column_name="ap_lo")

df1_metadata = {
    "fields": {
        "case_id": {"type": "numerical", "subtype": "integer"},
        "Hospital_code": {"type": "categorical"},
        "Hospital_type_code": {"type": "categorical"},
        "City_Code_Hospital": {"type": "categorical"},
        "Hospital_region_code": {"type": "categorical"},
        "Available Extra Rooms in Hospital": {"type": "numerical", "subtype": "integer"},
        "Department": {"type": "categorical"},
        "Ward_Type": {"type": "categorical"},
        "Ward_Facility_Code": {"type": "categorical"},
        "Bed Grade": {"type": "categorical"},
        "patientid": {"type": "numerical", "subtype": "integer"},
        "City_Code_Patient": {"type": "categorical"},
        "Type of Admission": {"type": "categorical"},
        "Severity of Illness": {"type": "categorical"},
        "Visitors with Patient": {"type": "numerical", "subtype": "integer"},
        "Age": {"type": "categorical"},
        "Admission_Deposit": {"type": "numerical", "subtype": "integer"},
        "LOS": {"type": "numerical", "subtype": "integer"},
    },
    "constraints": [df1_los_rounding, df1_fixed1],
    "primary_key": "case_id"
}

df2_metadata = {
    "fields": {
        "id": {"type": "numerical", "subtype": "integer"},
        "age": {"type": "numerical", "subtype": "integer"},
        "gender": {"type": "numerical", "subtype": "integer"},
        "height": {"type": "numerical", "subtype": "integer"},
        "weight": {"type": "numerical", "subtype": "integer"},
        "ap_hi": {"type": "numerical", "subtype": "integer"},
        "ap_lo": {"type": "numerical", "subtype": "integer"},
        "cholesterol": {"type": "categorical"},
        "gluc": {"type": "categorical"},
        "smoke": {"type": "categorical"},
        "alco": {"type": "categorical"},
        "active": {"type": "categorical"},
        "cardio": {"type": "categorical"},
    },
    "constraints": [df2_ap_inequality],
    "primary_key": "id"
}

df3_metadata = {
    "fields": {
        "id": {"type": "numerical", "subtype": "integer"},
        "gender": {"type": "categorical"},
        "age": {"type": "numerical", "subtype": "integer"},
        "hypertension": {"type": "categorical"},
        "heart_disease": {"type": "categorical"},
        "ever_married": {"type": "categorical"},
        "work_type": {"type": "categorical"},
        "Residence_type": {"type": "categorical"},
        "avg_glucose_level": {"type": "numerical", "subtype": "float"},
        "bmi": {"type": "numerical", "subtype": "float"},
        "smoking_status": {"type": "categorical"},
        "stroke": {"type": "categorical"},
    },
    "constraints": [],
    "primary_key": "id"
}

def generate(type, data, metadata, rows, path, filename):
    if type == "SDV":
        model = SDV()
    if type == "CTGAN":
        model = CTGAN(table_metadata=metadata)
    if type == "TabularPreset":
        model = TabularPreset(name='FAST_ML', metadata=metadata)
    if type == "GaussianCopula":
        model = GaussianCopula(table_metadata=metadata)
    if type == "TVAE":
        model = TVAE(table_metadata=metadata)

    print("Model created")
    model.fit(data)
    print("Model fitted")
    samples = model.sample(num_rows=rows)
    print("Model sampled")
    samples.to_csv(path + filename, index=False)
    print("Data saved")
