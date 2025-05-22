import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

scaler = None


def scaleData(X_train, X_test):
    global scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(BASE_DIR, "preprocess", "anemia", "scaler.pkl")


    joblib.dump(scaler, data_path)

    return X_train_scaled, X_test_scaled


def changeLabels():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(BASE_DIR, "data", "anemia", "dane-anemia-synthetic.csv")
    df = pd.read_csv(data_path)


    label_mapping = {
        "Anemia Mikrocytarna": 0,
        "Anemia Hemolityczna": 1,
        "Anemia Aplastyczna": 2,
        "Anemia Normocytarna": 3,
        "Healthy": 4
    }
    df['Label_num'] = df['label'].map(label_mapping)
    return df


if __name__ == "__main__":
    df_test = changeLabels()

    print(df_test['label'].value_counts())

    print(df_test.shape)
    print(df_test.head())
