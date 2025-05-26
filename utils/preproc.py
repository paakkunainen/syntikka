import pandas as pd
import numpy as np
import random
import torch
from sklearn.preprocessing import StandardScaler
from rdt.transformers import numerical, categorical
from typing import List, Dict


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def transform_data(
    original_data: pd.DataFrame,
    cat_cols: List[str],
    cont_cols: List[str],
    pre_proc_method: str = "GMM",
):
    original_categorical_columns = cat_cols.copy()
    original_continuous_columns = cont_cols.copy()

    categorical_columns = original_categorical_columns.copy()
    continuous_columns = original_continuous_columns.copy()

    continuous_transformers: Dict[str, object] = {}
    categorical_transformers: Dict[str, object] = {}

    num_categories: List[int] = []
    for col in original_categorical_columns:
        if original_data[col].dtype == float:
            original_data[col] = original_data[col].astype(int)
        if original_data[col].dtype == int or np.issubdtype(
            original_data[col].dtype, np.integer
        ):
            num_categories.append(int(np.amax(original_data[col]) + 1))
        else:
            num_categories.append(original_data[col].astype(str).nunique())

    transformed_dataset = original_data.copy(deep=True)

    if pre_proc_method == "GMM":
        for column in continuous_columns:
            normalizer = numerical.ClusterBasedNormalizer()
            normalizer.fit(transformed_dataset, column)
            continuous_transformers[f"continuous_{column}"] = normalizer
            transformed_dataset = normalizer.transform(transformed_dataset)
            categorical_columns.append(f"{column}.component")
        continuous_columns = [f"{col}.normalized" for col in continuous_columns]
    else:
        for column in continuous_columns:
            scaler = StandardScaler()
            transformed_dataset[column] = scaler.fit_transform(
                transformed_dataset[[column]].values
            )
            continuous_transformers[f"continuous_{column}"] = scaler

    num_categories = []
    for col in categorical_columns:
        if transformed_dataset[col].dtype == float:
            transformed_dataset[col] = transformed_dataset[col].astype(int)
        if np.issubdtype(transformed_dataset[col].dtype, np.integer):
            num_categories.append(int(np.amax(transformed_dataset[col]) + 1))
        else:
            num_categories.append(transformed_dataset[col].astype(str).nunique())

    for column in categorical_columns:
        ohe = categorical.OneHotEncoder()
        ohe.fit(transformed_dataset, column)
        categorical_transformers[f"categorical_{column}"] = ohe
        transformed_dataset = ohe.transform(transformed_dataset)

    reordered_dataframe = pd.concat(
        [
            transformed_dataset.iloc[:, len(continuous_columns) :],
            transformed_dataset.iloc[:, : len(continuous_columns)],
        ],
        axis=1,
    )

    x_train = reordered_dataframe.to_numpy(dtype="float32")

    return (
        x_train,
        original_data,
        reordered_dataframe.columns,
        continuous_transformers,
        categorical_transformers,
        num_categories,
        len(continuous_columns),
    )


def reverse_transformers(
    synthetic_data,
    original_data_columns,
    cont_transformers=None,
    cat_transformers=None,
    pre_proc_method="GMM",
):
    synthetic_transformed_data = synthetic_data.copy(deep=True)

    if cat_transformers is not None:
        for transformer_name in cat_transformers:
            transformer = cat_transformers[transformer_name]
            column_name = transformer_name[12:]
            synthetic_transformed_data = transformer.reverse_transform(
                synthetic_transformed_data
            )

    if cont_transformers is not None:
        if pre_proc_method == "GMM":
            for transformer_name in cont_transformers:
                transformer = cont_transformers[transformer_name]
                column_name = transformer_name[11:]
                synthetic_transformed_data = transformer.reverse_transform(
                    synthetic_transformed_data
                )
        elif pre_proc_method == "standard":
            for transformer_name in cont_transformers:
                transformer = cont_transformers[transformer_name]
                column_name = transformer_name[11:]
                # Reverse the standard scaling
                synthetic_transformed_data[column_name] = transformer.inverse_transform(
                    synthetic_transformed_data[column_name].values.reshape(-1, 1)
                ).flatten()

    synthetic_transformed_data = pd.DataFrame(
        synthetic_transformed_data, columns=original_data_columns
    )

    return synthetic_transformed_data


def unix_to_datetime(df, column_name):
    df[column_name] = pd.to_datetime(df[f"{column_name}"] * 1e11, unit="ns")
