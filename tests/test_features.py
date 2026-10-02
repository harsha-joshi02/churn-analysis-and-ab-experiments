import pandas as pd

from src.features import load_raw_data


def test_gender_encoding_preserves_male_and_female(tmp_path):
    source = pd.DataFrame(
        {
            "gender": ["Male", "Female"],
            "MonthlyCharges": [50.0, 60.0],
            "TotalCharges": [100.0, 120.0],
            "Churn": ["No", "Yes"],
        }
    )
    path = tmp_path / "customers.csv"
    source.to_csv(path, index=False)

    loaded = load_raw_data(path)

    assert loaded["gender"].tolist() == [1, 0]
    assert loaded["churn"].tolist() == [0, 1]
