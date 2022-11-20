import pandas as pd
import numpy as np
import xgboost
from sdv import SDV, Metadata
from sdv.evaluation import evaluate
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sdv.tabular import CTGAN

np.random.seed(5)

df1_300 = pd.read_csv("xgboost/Data/Dataset1/train_data.csv")
df1_50 = pd.read_csv("xgboost/Data/Dataset1/original_50.csv")
df1_150 = pd.read_csv("xgboost/Data/Dataset1/original_150.csv")

df1_sdv_300 = pd.read_csv("xgboost/Data/Dataset1/syn_sdv_300.csv")
df1_sdv_150 = pd.read_csv("xgboost/Data/Dataset1/syn_sdv_150.csv")
df1_sdv_50 = pd.read_csv("xgboost/Data/Dataset1/syn_sdv_50.csv")

df1_sdv_300_CTGAN = pd.read_csv("xgboost/Data/Dataset1/syn_sdv_300_CTGAN.csv")
df1_sdv_300_TabularPreset = pd.read_csv("xgboost/Data/Dataset1/syn_sdv_300_TabularPreset.csv")
df1_sdv_300_GaussianCopula = pd.read_csv("xgboost/Data/Dataset1/syn_sdv_300_GaussianCopula.csv")
df1_sdv_300_TVAE = pd.read_csv("xgboost/Data/Dataset1/syn_sdv_300_TVAE.csv")
df1_sdv_300_TVAE2 = pd.read_csv("xgboost/Data/Dataset1/syn_sdv_300_TVAE2.csv")

df1_gretel_300 = pd.read_csv("xgboost/Data/Dataset1/syn_gretel_300.csv")
df1_gretel_300[['Stay','LOS']] = df1_gretel_300.Stay.str.split('-',expand=True)
del df1_gretel_300["Stay"]
df1_gretel_300["LOS"] = df1_gretel_300["LOS"].fillna(110)
df1_gretel_300["LOS"] = df1_gretel_300["LOS"].astype(int)

df1_gretel_300_highaccuracy = pd.read_csv("xgboost/Data/Dataset1/syn_gretel_300_highaccuracy.csv")


df2_70 = pd.read_csv("xgboost/Data/Dataset2/cardio_train.csv")
df2_10 = pd.read_csv("xgboost/Data/Dataset2/original_10.csv")
df2_35 = pd.read_csv("xgboost/Data/Dataset2/original_35.csv")

df2_sdv_70 = pd.read_csv("xgboost/Data/Dataset2/syn_sdv_70.csv")
df2_sdv_35 = pd.read_csv("xgboost/Data/Dataset2/syn_sdv_35.csv")
df2_sdv_10 = pd.read_csv("xgboost/Data/Dataset2/syn_sdv_10.csv")

df2_sdv_70_CTGAN = pd.read_csv("xgboost/Data/Dataset2/syn_sdv_70_CTGAN.csv")
df2_sdv_70_CTGAN2 = pd.read_csv("xgboost/Data/Dataset2/syn_sdv_70_CTGAN2.csv")
df2_sdv_70_TabularPreset = pd.read_csv("xgboost/Data/Dataset2/syn_sdv_70_TabularPreset.csv")
df2_sdv_70_TabularPreset2 = pd.read_csv("xgboost/Data/Dataset2/syn_sdv_70_TabularPreset2.csv")
df2_sdv_70_GaussianCopula = pd.read_csv("xgboost/Data/Dataset2/syn_sdv_70_GaussianCopula.csv")
df2_sdv_70_GaussianCopula2 = pd.read_csv("xgboost/Data/Dataset2/syn_sdv_70_GaussianCopula2.csv")
df2_sdv_70_TVAE = pd.read_csv("xgboost/Data/Dataset2/syn_sdv_70_TVAE.csv")
df2_sdv_70_TVAE2 = pd.read_csv("xgboost/Data/Dataset2/syn_sdv_70_TVAE2.csv")

df2_gretel_70 = pd.read_csv("xgboost/Data/Dataset2/syn_gretel_70.csv")
df2_gretel_70_highaccuracy = pd.read_csv("xgboost/Data/Dataset2/syn_gretel_70_highaccuracy.csv")


df3_5 = pd.read_csv("xgboost/Data/Dataset3/healthcare-dataset-stroke-data.csv")
df3_1 = pd.read_csv("xgboost/Data/Dataset3/original_1.csv")

df3_sdv_5 = pd.read_csv("xgboost/Data/Dataset3/syn_sdv_5.csv")
df3_sdv_1 = pd.read_csv("xgboost/Data/Dataset3/syn_sdv_1.csv")

df3_sdv_5_CTGAN = pd.read_csv("xgboost/Data/Dataset3/syn_sdv_5_CTGAN.csv")
df3_sdv_5_TabularPreset = pd.read_csv("xgboost/Data/Dataset3/syn_sdv_5_TabularPreset.csv")
df3_sdv_5_GaussianCopula = pd.read_csv("xgboost/Data/Dataset3/syn_sdv_5_GaussianCopula.csv")
df3_sdv_5_TVAE = pd.read_csv("xgboost/Data/Dataset3/syn_sdv_5_TVAE.csv")

df3_gretel_5 = pd.read_csv("xgboost/Data/Dataset3/syn_gretel1.csv")
df3_gretel_5_highaccuracy = pd.read_csv("xgboost/Data/Dataset3/syn_gretel_highaccuracy.csv")
df3_gretel_5_complex = pd.read_csv("xgboost/Data/Dataset3/syn_gretel_complex.csv")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

lbl = preprocessing.LabelEncoder()

def syn_trained_1(data, name, original=None):
    if original is None:
        data_combined1_300 = data.copy()
        data_combined1_300 = data_combined1_300.drop(["case_id", "patientid"], axis=1)

        for index, row in data_combined1_300.iterrows():
            data_combined1_300.at[index, "LOS"] = (row["LOS"] - 10) / 10

        data_X1_300 = data_combined1_300.iloc[:,:-1]
        data_y1_300 = data_combined1_300.iloc[:,-1:]

        lbl = preprocessing.LabelEncoder()
        data_X1_300['Hospital_type_code'] = lbl.fit_transform(data_X1_300['Hospital_type_code'].astype(str))
        data_X1_300['Hospital_region_code'] = lbl.fit_transform(data_X1_300['Hospital_region_code'].astype(str))
        data_X1_300['Department'] = lbl.fit_transform(data_X1_300['Department'].astype(str))
        data_X1_300['Ward_Type'] = lbl.fit_transform(data_X1_300['Ward_Type'].astype(str))
        data_X1_300['Ward_Facility_Code'] = lbl.fit_transform(data_X1_300['Ward_Facility_Code'].astype(str))
        data_X1_300['Type of Admission'] = lbl.fit_transform(data_X1_300['Type of Admission'].astype(str))
        data_X1_300['Severity of Illness'] = lbl.fit_transform(data_X1_300['Severity of Illness'].astype(str))
        data_X1_300['Age'] = lbl.fit_transform(data_X1_300['Age'].astype(str))

        X_train1_300, X_test1_300, y_train1_300, y_test1_300 =\
            train_test_split(data_X1_300, data_y1_300, test_size=0.2, random_state=1)
        model1_300 = xgboost.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        model1_300.fit(X_train1_300, y_train1_300)
        y_pred1_300 = model1_300.predict(X_test1_300)
        print("Dataset 1", name, (accuracy_score(y_test1_300, y_pred1_300)*100))

    else:
        data_combined1_300_original = original.copy()
        data_combined1_300_original = data_combined1_300_original.drop(["case_id", "patientid"], axis=1)

        for index, row in data_combined1_300_original.iterrows():
            data_combined1_300_original.at[index, "LOS"] = (row["LOS"] - 10) / 10

        data_X1_300_original = data_combined1_300_original.iloc[:, :-1]
        data_y1_300_original = data_combined1_300_original.iloc[:, -1:]

        lbl = preprocessing.LabelEncoder()
        data_X1_300_original['Hospital_type_code'] = lbl.fit_transform(data_X1_300_original['Hospital_type_code'].astype(str))
        data_X1_300_original['Hospital_region_code'] = lbl.fit_transform(data_X1_300_original['Hospital_region_code'].astype(str))
        data_X1_300_original['Department'] = lbl.fit_transform(data_X1_300_original['Department'].astype(str))
        data_X1_300_original['Ward_Type'] = lbl.fit_transform(data_X1_300_original['Ward_Type'].astype(str))
        data_X1_300_original['Ward_Facility_Code'] = lbl.fit_transform(data_X1_300_original['Ward_Facility_Code'].astype(str))
        data_X1_300_original['Type of Admission'] = lbl.fit_transform(data_X1_300_original['Type of Admission'].astype(str))
        data_X1_300_original['Severity of Illness'] = lbl.fit_transform(data_X1_300_original['Severity of Illness'].astype(str))
        data_X1_300_original['Age'] = lbl.fit_transform(data_X1_300_original['Age'].astype(str))

        X_train1_300_original, X_test1_300_original, y_train1_300_original, y_test1_300_original = \
            train_test_split(data_X1_300_original, data_y1_300_original, test_size=0.2, random_state=1)

        data_combined1_300 = data.copy()
        data_combined1_300 = data_combined1_300.drop(["case_id", "patientid"], axis=1)

        for index, row in data_combined1_300.iterrows():
            data_combined1_300.at[index, "LOS"] = (row["LOS"] - 10) / 10

        data_X1_300 = data_combined1_300.iloc[:, :-1]
        data_y1_300 = data_combined1_300.iloc[:, -1:]

        lbl = preprocessing.LabelEncoder()
        data_X1_300['Hospital_type_code'] = lbl.fit_transform(data_X1_300['Hospital_type_code'].astype(str))
        data_X1_300['Hospital_region_code'] = lbl.fit_transform(data_X1_300['Hospital_region_code'].astype(str))
        data_X1_300['Department'] = lbl.fit_transform(data_X1_300['Department'].astype(str))
        data_X1_300['Ward_Type'] = lbl.fit_transform(data_X1_300['Ward_Type'].astype(str))
        data_X1_300['Ward_Facility_Code'] = lbl.fit_transform(data_X1_300['Ward_Facility_Code'].astype(str))
        data_X1_300['Type of Admission'] = lbl.fit_transform(data_X1_300['Type of Admission'].astype(str))
        data_X1_300['Severity of Illness'] = lbl.fit_transform(data_X1_300['Severity of Illness'].astype(str))
        data_X1_300['Age'] = lbl.fit_transform(data_X1_300['Age'].astype(str))

        X_train1_300, X_test1_300, y_train1_300, y_test1_300 = \
            train_test_split(data_X1_300, data_y1_300, test_size=0.2, random_state=1)

        model1_300 = xgboost.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        model1_300.fit(X_train1_300, y_train1_300)
        y_pred1_300 = model1_300.predict(X_test1_300_original)
        print("Dataset 1", name, (accuracy_score(y_test1_300_original, y_pred1_300)*100))


def syn_trained_2(data, name, original=None):
    if original is None:
        data_combined2_70 = data.copy()
        data_combined2_70 = data_combined2_70.drop(["id"], axis=1)

        data_X2_70 = data_combined2_70.iloc[:, :-1]
        data_y2_70 = data_combined2_70.iloc[:, -1:]

        X_train2_70, X_test2_70, y_train2_70, y_test2_70 = \
            train_test_split(data_X2_70, data_y2_70, test_size=0.2, random_state=1)
        model2_70 = xgboost.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        model2_70.fit(X_train2_70, y_train2_70)
        y_pred2_70 = model2_70.predict(X_test2_70)
        print("Dataset 2", name, (accuracy_score(y_test2_70, y_pred2_70)*100))
    else:
        data_combined2_70_original = original.copy()
        data_combined2_70_original = data_combined2_70_original.drop(["id"], axis=1)

        data_X2_70_original = data_combined2_70_original.iloc[:, :-1]
        data_y2_70_original = data_combined2_70_original.iloc[:, -1:]

        X_train2_70_original, X_test2_70_original, y_train2_70_original, y_test2_70_original = \
            train_test_split(data_X2_70_original, data_y2_70_original, test_size=0.2, random_state=1)

        data_combined2_70 = data.copy()
        data_combined2_70 = data_combined2_70.drop(["id"], axis=1)

        data_X2_70 = data_combined2_70.iloc[:, :-1]
        data_y2_70 = data_combined2_70.iloc[:, -1:]

        X_train2_70, X_test2_70, y_train2_70, y_test2_70 = \
            train_test_split(data_X2_70, data_y2_70, test_size=0.2, random_state=1)
        model2_70 = xgboost.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        model2_70.fit(X_train2_70, y_train2_70)
        y_pred2_70 = model2_70.predict(X_test2_70_original)
        print("Dataset 2", name, (accuracy_score(y_test2_70_original, y_pred2_70)*100))


def syn_trained_2_new(data, name, original):
    data_combined2_70_original = original.copy()
    data_combined2_70_original = data_combined2_70_original.drop(["id"], axis=1)

    data_X2_70_original = data_combined2_70_original.iloc[:, :-1]
    data_y2_70_original = data_combined2_70_original.iloc[:, -1:]

    X_train2_70_original, X_test2_70_original, y_train2_70_original, y_test2_70_original = \
        train_test_split(data_X2_70_original, data_y2_70_original, test_size=0.2, random_state=1)

    data_combined2_70 = data.copy()
    data_combined2_70 = data_combined2_70.drop(["id"], axis=1)

    data_X2_70 = data_combined2_70.iloc[:, :-1]
    data_y2_70 = data_combined2_70.iloc[:, -1:]

    X_train2_70, X_test2_70, y_train2_70, y_test2_70 = \
        train_test_split(data_X2_70, data_y2_70, test_size=0.2, random_state=1)

    X_train2_70_new = pd.concat([X_train2_70, X_train2_70_original])
    y_train2_70_new = pd.concat([y_train2_70, y_train2_70_original])

    model2_70 = xgboost.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model2_70.fit(X_train2_70_new, y_train2_70_new)
    y_pred2_70 = model2_70.predict(X_test2_70_original)
    print("Dataset 2", name, (accuracy_score(y_test2_70_original, y_pred2_70)*100))


def syn_trained_3(data, name, original=None):
    if original is None:
        data_combined3_5 = data.copy()
        data_combined3_5 = data_combined3_5.drop(["id"], axis=1)

        data_X3_5 = data_combined3_5.iloc[:, :-1]
        data_y3_5 = data_combined3_5.iloc[:, -1:]

        data_X3_5['gender'] = lbl.fit_transform(data_X3_5['gender'].astype(str))
        data_X3_5['ever_married'] = lbl.fit_transform(data_X3_5['ever_married'].astype(str))
        data_X3_5['work_type'] = lbl.fit_transform(data_X3_5['work_type'].astype(str))
        data_X3_5['Residence_type'] = lbl.fit_transform(data_X3_5['Residence_type'].astype(str))
        data_X3_5['smoking_status'] = lbl.fit_transform(data_X3_5['smoking_status'].astype(str))

        X_train3_5, X_test3_5, y_train3_5, y_test3_5 = \
            train_test_split(data_X3_5, data_y3_5, test_size=0.2, random_state=1)
        model3_5 = xgboost.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        model3_5.fit(X_train3_5, y_train3_5)
        y_pred3_5 = model3_5.predict(X_test3_5)
        print("Dataset 3", name, (accuracy_score(y_test3_5, y_pred3_5)*100))
    else:
        data_combined3_5 = data.copy()
        data_combined3_5 = data_combined3_5.drop(["id"], axis=1)

        data_X3_5 = data_combined3_5.iloc[:, :-1]
        data_y3_5 = data_combined3_5.iloc[:, -1:]

        data_X3_5['gender'] = lbl.fit_transform(data_X3_5['gender'].astype(str))
        data_X3_5['ever_married'] = lbl.fit_transform(data_X3_5['ever_married'].astype(str))
        data_X3_5['work_type'] = lbl.fit_transform(data_X3_5['work_type'].astype(str))
        data_X3_5['Residence_type'] = lbl.fit_transform(data_X3_5['Residence_type'].astype(str))
        data_X3_5['smoking_status'] = lbl.fit_transform(data_X3_5['smoking_status'].astype(str))

        X_train3_5, X_test3_5, y_train3_5, y_test3_5 = \
            train_test_split(data_X3_5, data_y3_5, test_size=0.2, random_state=1)

        data_combined3_5_original = original.copy()
        data_combined3_5_original = data_combined3_5_original.drop(["id"], axis=1)

        data_X3_5_original = data_combined3_5_original.iloc[:, :-1]
        data_y3_5_original = data_combined3_5_original.iloc[:, -1:]

        data_X3_5_original['gender'] = lbl.fit_transform(data_X3_5_original['gender'].astype(str))
        data_X3_5_original['ever_married'] = lbl.fit_transform(data_X3_5_original['ever_married'].astype(str))
        data_X3_5_original['work_type'] = lbl.fit_transform(data_X3_5_original['work_type'].astype(str))
        data_X3_5_original['Residence_type'] = lbl.fit_transform(data_X3_5_original['Residence_type'].astype(str))
        data_X3_5_original['smoking_status'] = lbl.fit_transform(data_X3_5_original['smoking_status'].astype(str))

        X_train3_5_original, X_test3_5_original, y_train3_5_original, y_test3_5_original = \
            train_test_split(data_X3_5_original, data_y3_5_original, test_size=0.2, random_state=1)
        model3_5 = xgboost.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        model3_5.fit(X_train3_5, y_train3_5)
        y_pred3_5 = model3_5.predict(X_test3_5_original)
        print("Dataset 3", name, (accuracy_score(y_test3_5_original, y_pred3_5)*100))


def syn_trained_3_new(data, name, original):
    data_combined3_5 = data.copy()
    data_combined3_5 = data_combined3_5.drop(["id"], axis=1)

    data_X3_5 = data_combined3_5.iloc[:, :-1]
    data_y3_5 = data_combined3_5.iloc[:, -1:]

    data_X3_5['gender'] = lbl.fit_transform(data_X3_5['gender'].astype(str))
    data_X3_5['ever_married'] = lbl.fit_transform(data_X3_5['ever_married'].astype(str))
    data_X3_5['work_type'] = lbl.fit_transform(data_X3_5['work_type'].astype(str))
    data_X3_5['Residence_type'] = lbl.fit_transform(data_X3_5['Residence_type'].astype(str))
    data_X3_5['smoking_status'] = lbl.fit_transform(data_X3_5['smoking_status'].astype(str))

    X_train3_5, X_test3_5, y_train3_5, y_test3_5 = \
        train_test_split(data_X3_5, data_y3_5, test_size=0.2, random_state=1)

    data_combined3_5_original = original.copy()
    data_combined3_5_original = data_combined3_5_original.drop(["id"], axis=1)

    data_X3_5_original = data_combined3_5_original.iloc[:, :-1]
    data_y3_5_original = data_combined3_5_original.iloc[:, -1:]

    data_X3_5_original['gender'] = lbl.fit_transform(data_X3_5_original['gender'].astype(str))
    data_X3_5_original['ever_married'] = lbl.fit_transform(data_X3_5_original['ever_married'].astype(str))
    data_X3_5_original['work_type'] = lbl.fit_transform(data_X3_5_original['work_type'].astype(str))
    data_X3_5_original['Residence_type'] = lbl.fit_transform(data_X3_5_original['Residence_type'].astype(str))
    data_X3_5_original['smoking_status'] = lbl.fit_transform(data_X3_5_original['smoking_status'].astype(str))

    X_train3_5_original, X_test3_5_original, y_train3_5_original, y_test3_5_original = \
        train_test_split(data_X3_5_original, data_y3_5_original, test_size=0.2, random_state=1)

    X_train3_5_new = pd.concat([X_train3_5, X_train3_5_original])
    y_train3_5_new = pd.concat([y_train3_5, y_train3_5_original])

    model3_5 = xgboost.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model3_5.fit(X_train3_5_new, y_train3_5_new)
    y_pred3_5 = model3_5.predict(X_test3_5_original)
    print("Dataset 3", name, (accuracy_score(y_test3_5_original, y_pred3_5)*100))

def check1():
    syn_trained_1(df1_300, "Original:")

    syn_trained_1(df1_sdv_300, "SDV:")
    syn_trained_1(df1_sdv_300_CTGAN, "SDV CTGAN:")
    syn_trained_1(df1_sdv_300_TabularPreset, "SDV TabularPreset:")
    syn_trained_1(df1_sdv_300_GaussianCopula, "SDV GaussianCopula:")
    syn_trained_1(df1_sdv_300_TVAE, "SDV TVAE:")

    syn_trained_1(df1_gretel_300, "gretel:")
    syn_trained_1(df1_gretel_300_highaccuracy, "gretel highaccuracy:")
    print()


def check2():
    syn_trained_2(df2_70, "Original:")

    syn_trained_2(df2_sdv_70, "SDV:")
    syn_trained_2(df2_sdv_70_CTGAN2, "SDV CTGAN:")
    syn_trained_2(df2_sdv_70_TabularPreset2, "SDV TabularPreset:")
    syn_trained_2(df2_sdv_70_GaussianCopula2, "SDV GaussianCopula:")
    syn_trained_2(df2_sdv_70_TVAE2, "SDV TVAE:")

    syn_trained_2(df2_gretel_70, "gretel:")
    syn_trained_2(df2_gretel_70_highaccuracy, "gretel highaccuracy:")
    print()

def check3():
    syn_trained_3(df3_5, "Original:")

    syn_trained_3(df3_sdv_5, "SDV:")
    syn_trained_3(df3_sdv_5_CTGAN, "SDV CTGAN:")
    syn_trained_3(df3_sdv_5_TabularPreset, "SDV TabularPreset:")
    syn_trained_3(df3_sdv_5_GaussianCopula, "SDV GaussianCopula:")
    syn_trained_3(df3_sdv_5_TVAE, "SDV TVAE:")

    syn_trained_3(df3_gretel_5, "Gretel:")
    syn_trained_3(df3_gretel_5_highaccuracy, "Gretel highaccuracy:")
    print()

def check1_original():
    syn_trained_1(df1_300, "Original:")

    syn_trained_1(df1_sdv_300, "SDV:", df1_300)
    syn_trained_1(df1_sdv_300_CTGAN, "SDV CTGAN:", df1_300)
    syn_trained_1(df1_sdv_300_TabularPreset, "SDV TabularPreset:", df1_300)
    syn_trained_1(df1_sdv_300_GaussianCopula, "SDV GaussianCopula:", df1_300)
    syn_trained_1(df1_sdv_300_TVAE, "SDV TVAE:", df1_300)

    syn_trained_1(df1_gretel_300, "gretel:", df1_300)
    syn_trained_1(df1_gretel_300_highaccuracy, "gretel highaccuracy:", df1_300)
    print()


def check2_original():
    syn_trained_2(df2_70, "Original:")

    syn_trained_2(df2_sdv_70, "SDV:", df2_70)
    syn_trained_2(df2_sdv_70_CTGAN2, "SDV CTGAN:", df2_70)
    syn_trained_2(df2_sdv_70_TabularPreset2, "SDV TabularPreset:", df2_70)
    syn_trained_2(df2_sdv_70_GaussianCopula2, "SDV GaussianCopula:", df2_70)
    syn_trained_2(df2_sdv_70_TVAE2, "SDV TVAE:", df2_70)

    syn_trained_2(df2_gretel_70, "gretel:", df2_70)
    syn_trained_2(df2_gretel_70_highaccuracy, "gretel highaccuracy:", df2_70)
    print()

def check3_original():
    syn_trained_3(df3_5, "Original:")

    syn_trained_3(df3_sdv_5, "SDV:", df3_5)
    syn_trained_3(df3_sdv_5_CTGAN, "SDV CTGAN:", df3_5)
    syn_trained_3(df3_sdv_5_TabularPreset, "SDV TabularPreset:", df3_5)
    syn_trained_3(df3_sdv_5_GaussianCopula, "SDV GaussianCopula:", df3_5)
    syn_trained_3(df3_sdv_5_TVAE, "SDV TVAE:", df3_5)

    syn_trained_3(df3_gretel_5, "Gretel:", df3_5)
    syn_trained_3(df3_gretel_5_highaccuracy, "Gretel highaccuracy:", df3_5)
    print()

if __name__ == "__main__":
    print("Trained and tested on same model")
    check1()
    check2()
    check3()
    
    print("Trained on model, tested on original")
    check1_original()
    check2_original()
    check3_original()
    