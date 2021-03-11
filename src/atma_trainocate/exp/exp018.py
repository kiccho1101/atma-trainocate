# %%
import colorsys
import math
import os
import pickle
import random
import re
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import mlflow.lightgbm
import numpy as np
import optuna
import optuna.integration.lightgbm as lgbo
import pandas as pd
import psutil
import pycld2 as cld2
import seaborn as sns
import torch
import torch.backends
import torch.backends.cudnn
import torch.cuda
import umap
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
from fasttext import load_model
from optuna.integration.lightgbm import LightGBMTunerCV
from optuna.trial import Trial
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from stop_words import get_stop_words
from tqdm.autonotebook import tqdm


class GlobalUtil:
    @staticmethod
    def seed_everything(seed: int):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


class TimeUtil:
    @staticmethod
    @contextmanager
    def timer(name: str):
        t0 = time.time()
        p = psutil.Process(os.getpid())
        m0 = p.memory_info()[0] / 2.0 ** 30
        p0 = psutil.virtual_memory().percent
        print(f"[{name}] start [{m0:.1f}GB({p0:.1f}%)]")
        yield
        m1 = p.memory_info()[0] / 2.0 ** 30
        p1 = psutil.virtual_memory().percent
        print(f"[{name}] start [{m0:.1f}GB({p0:.1f}%)]")
        delta = m1 - m0
        sign = "+" if delta >= 0 else "-"
        delta = math.fabs(delta)
        print(
            f"[{name}] done [{m1:.1f}GB({p1:.1f}%)({sign}{delta:.3f}GB)] {time.time() - t0:.4f} s"
        )
        print()


@dataclass
class RawData:
    train: pd.DataFrame
    test: pd.DataFrame
    color: pd.DataFrame
    historical_person: pd.DataFrame
    maker: pd.DataFrame
    material: pd.DataFrame
    object_collection: pd.DataFrame
    palette: pd.DataFrame
    principal_maker_occupation: pd.DataFrame
    principal_maker: pd.DataFrame
    production_place: pd.DataFrame
    technique: pd.DataFrame
    colorspace_selected_pca: pd.DataFrame
    sample_submission: pd.DataFrame


class Config:
    seed = 77
    n_splits = 5
    img_width = 512
    img_height = 512
    exp_name = "atma-cv"
    batch_size = 32
    num_workers = 4
    root_dir = Path.cwd().parents[2]
    data_dir = Path.cwd().parents[2] / "data/atmacup10_dataset"
    pca_method = "PCA"
    principal_maker_n_components = 10
    title_n_components = 10
    long_title_n_components = 15
    desc_en_n_components = 15
    desc_n_components = 15
    long_title_desc_en_n_components = 15
    color_n_components = 3
    material_n_components = 10
    technique_n_components = 5
    colorspace_n_components = 30
    title_maker_n_components = 15
    technique_col_num = 32
    object_collections = ["prints", "paintings", "other"]
    cat_aggs = ["count"]
    # bert_model = "_stsb_roberta_large"
    # bert_model = "_bert_base_multilingual_uncase"
    # bert_model = "_bert_base_multilingual_case"
    bert_model = "_bert_base_multilingual_case_mean"
    # bert_model = ""
    lgb_params = {
        "num_leaves": 100,
        "min_data_in_leaf": 20,
        "objective": "regression",
        "max_depth": -1,
        "learning_rate": 0.02,
        "bagging_freq": 1,
        "bagging_fraction": 0.8,
        "bagging_seed": seed,
        "verbosity": -1,
        "reg_alpha": 0.1,
        "reg_lambda": 0.3,
        "colsample_bytree": 0.7,
        "metric": "rmse",
        "num_threads": -1,
    }
    # cat_params = {
    #     "depth": 9,
    #     "learning_rate": 0.026159441983418394,
    #     "od_type": "IncToDec",
    # }
    # cat_params = {
    #     "num_leaves": 89,
    #     "min_data_in_leaf": 20,
    #     "grow_policy": "Lossguide",
    #     "learning_rate": 0.025,
    #     "reg_lambda": 0.3,
    # }
    cat_params = {
        "depth": 16,
        "num_leaves": 130,
        "min_data_in_leaf": 22,
        "learning_rate": 0.01734679190329074,
        "reg_lambda": 0.3467458599178205,
        "grow_policy": "Lossguide",
        "random_seed": seed,
    }
    cat_params = {
        "grow_policy": "Lossguide",
        "random_seed": seed,
        "depth": 20,
        "num_leaves": 250,
        "min_data_in_leaf": 28,
        "learning_rate": 0.0051686369060981,
        "reg_lambda": 0.3,
    }
    xgb_params = {
        "num_leaves": 89,
        "min_data_in_leaf": 20,
        "learning_rate": 0.03,
        "bagging_freq": 1,
        "bagging_fraction": 0.8,
        "bagging_seed": seed,
        "reg_alpha": 0.1,
        "reg_lambda": 0.3,
        "colsample_bytree": 0.7,
        "metric": "rmse",
    }


{
    "depth": 30,
    "num_leaves": 146,
    "min_data_in_leaf": 48,
    "learning_rate": 0.013776138788920873,
    "reg_lambda": 0.24785444165604187,
}
{
    "depth": 14,
    "num_leaves": 200,
    "min_data_in_leaf": 7,
    "learning_rate": 0.01096658270026146,
    "reg_lambda": 0.03125474578559077,
}
{
    "depth": 20,
    "num_leaves": 138,
    "min_data_in_leaf": 20,
    "learning_rate": 0.010070282024956259,
    "reg_lambda": 0.012419909785704676,
}

{
    "depth": 27,
    "num_leaves": 250,
    "min_data_in_leaf": 28,
    "learning_rate": 0.0051686369060981,
    "reg_lambda": 0.4182381007114778,
}
{
    "depth": 17,
    "num_leaves": 231,
    "min_data_in_leaf": 20,
    "learning_rate": 0.005404462231314783,
    "reg_lambda": 0.07032144600604655,
}


class MlflowUtil:
    @staticmethod
    def start_run(models: str):
        name = input("name: ")
        if mlflow.active_run() is not None:
            mlflow.end_run()
        if mlflow.get_experiment_by_name(Config.exp_name) is None:
            mlflow.create_experiment(Config.exp_name)
        experiment_id = mlflow.get_experiment_by_name(Config.exp_name).experiment_id
        exp = str(Path(__file__)).split("/")[-1].replace(".py", "")
        mlflow.start_run(
            run_name=f"{exp}: ({models}) {name}", experiment_id=experiment_id
        )
        mlflow.log_param("exp", exp)

    @staticmethod
    def log_config():
        mlflow.log_param("seed", Config.seed)
        mlflow.log_param("n_splits", Config.n_splits)
        mlflow.log_param("bert_model", Config.bert_model)
        mlflow.log_params({f"lgb_{k}": v for k, v in Config.lgb_params.items()})
        mlflow.log_params({f"cat_{k}": v for k, v in Config.cat_params.items()})


def label_encoding(
    label_encs: Dict[str, LabelEncoder], raw: RawData, col: str
) -> Tuple[Dict[str, LabelEncoder], RawData]:
    raw.train[col] = raw.train[col].fillna("nan")
    raw.test[col] = raw.test[col].fillna("nan")
    label_encs[col].fit(list(set(raw.train[col].tolist() + raw.test[col].tolist())))
    raw.train[col] = label_encs[col].transform(raw.train[col])
    raw.test[col] = label_encs[col].transform(raw.test[col])
    return label_encs, raw


def cat_encoding(raw: RawData, col: str) -> RawData:
    exists = False
    for agg in Config.cat_aggs:
        if f"{col}_{agg}" in raw.train.columns:
            exists = True
    if exists:
        return raw
    _agg_df = raw.train.groupby(col)["likes_log"].agg(Config.cat_aggs)
    _agg_df.columns = [f"{col}_{agg}" for agg in Config.cat_aggs]
    raw.train = raw.train.merge(_agg_df, on=col, how="left")
    raw.test = raw.test.merge(_agg_df, on=col, how="left")

    for agg in Config.cat_aggs:
        raw.train[f"{col}_{agg}"] = raw.train[f"{col}_{agg}"].fillna(0)
        raw.test[f"{col}_{agg}"] = raw.test[f"{col}_{agg}"].fillna(0)
    return raw


def _fe_sub_title(df: pd.DataFrame) -> pd.DataFrame:
    for axis in ["h", "w", "t", "d"]:
        column_name = f"size_{axis}"
        size_info = df["sub_title"].str.extract(
            r"{} (\d*|\d*\.\d*)(cm|mm)".format(axis)
        )
        size_info = size_info.rename(columns={0: column_name, 1: "unit"})
        size_info[column_name] = (
            size_info[column_name].replace("", np.nan).astype(float)
        )
        size_info[column_name] = size_info.apply(
            lambda row: row[column_name] * 10
            if row["unit"] == "cm"
            else row[column_name],
            axis=1,
        )
        df[column_name] = size_info[column_name]
    df["size_area"] = df["size_h"] * df["size_w"]
    df["sub_title_len"] = df["sub_title"].str.len()
    df["title_len"] = df["title"].str.len()
    df["title_word_num"] = df["title"].map(lambda s: len(s.split()))
    df["title_capital_word_num"] = df["title"].map(
        lambda s: sum([1 if word[0].isupper() else 0 for word in s.split()])
    )
    df["title_contains_the"] = (
        df["title"].str.lower().str.contains("the").astype(np.int8)
    )
    df["description_en_len"] = df["description_en"].fillna("").str.len()
    df["description_en_word_num"] = (
        df["description_en"].fillna("").map(lambda s: len(s.split()))
    )
    df["size_hw_ratio"] = df["size_h"] / df["size_w"]
    df["more_title"] = df["more_title"].fillna("")
    df["more_title_is_less_than_title"] = df.apply(
        lambda row: len(row.more_title) < len(row.title), axis=1
    ).astype(np.int8)
    return df


def fe_sub_title(raw: RawData) -> RawData:
    raw.train = _fe_sub_title(raw.train)
    raw.test = _fe_sub_title(raw.test)
    return raw


def _fe_dating(df: pd.DataFrame) -> pd.DataFrame:
    df["dating_year_diff"] = df["dating_year_late"] - df["dating_year_early"]
    df["dating_acquisition_diff"] = (
        pd.to_datetime(df["acquisition_date"]).dt.year - df["dating_sorting_date"]
    )
    return df


def fe_dating(raw: RawData) -> RawData:
    raw.train = _fe_dating(raw.train)
    raw.test = _fe_dating(raw.test)
    return raw


def fe_colorspace_pca(raw: RawData) -> RawData:
    pca = PCA(n_components=Config.colorspace_n_components)
    reduced = pca.fit_transform(
        raw.colorspace_selected_pca.filter(like="color_").values
    )
    for i in range(Config.colorspace_n_components):
        raw.colorspace_selected_pca[f"colorspace_pca_{i}"] = reduced[:, i]
    _df = raw.colorspace_selected_pca[
        ["object_id"]
        + [f"colorspace_pca_{i}" for i in range(Config.colorspace_n_components)]
    ]
    raw.train = raw.train.merge(_df, on="object_id", how="left")
    raw.test = raw.test.merge(_df, on="object_id", how="left")
    return raw


def fe_color(raw: RawData) -> RawData:
    max_percentage = defaultdict(int)
    max_hex = defaultdict(str)

    for row in tqdm(raw.color.itertuples(), total=len(raw.color)):
        if max_percentage[row.object_id] < row.percentage:
            max_percentage[row.object_id] = row.percentage
            max_hex[row.object_id] = row.hex
    raw.train["max_percentage"] = raw.train["object_id"].map(max_percentage)
    raw.test["max_percentage"] = raw.test["object_id"].map(max_percentage)
    raw.train["max_hex"] = raw.train["object_id"].map(max_hex)
    raw.test["max_hex"] = raw.test["object_id"].map(max_hex)
    color_count = (
        raw.color[raw.color["percentage"] > 10]
        .groupby("object_id")["percentage"]
        .agg("count")
        .to_dict()
    )
    raw.train["color_count"] = raw.train["object_id"].map(color_count).fillna(0)
    raw.test["color_count"] = raw.test["object_id"].map(color_count).fillna(0)
    return raw


def fe_technique(raw: RawData) -> Tuple[RawData, int]:
    _counts = raw.technique["name"].value_counts()
    _dict = defaultdict(str)
    for row in tqdm(raw.technique.itertuples(), total=len(raw.technique)):
        if (
            row.object_id not in _dict
            or _counts[_dict[row.object_id]] < _counts[row.name]
        ):
            _dict[row.object_id] = row.name
    raw.train["technique"] = raw.train["object_id"].map(_dict)
    raw.test["technique"] = raw.test["object_id"].map(_dict)

    raw.technique["name_"] = raw.technique["name"].str.replace(" ", "")
    _df = (
        raw.technique.groupby("object_id")["name_"]
        .apply(list)
        .map(lambda l: " ".join(l))
        .reset_index()
    )

    tfidf = TfidfVectorizer(min_df=1)
    tfidf.fit(_df["name_"].tolist())
    mat = tfidf.transform(_df["name_"]).todense()

    col_num = mat.shape[1]
    columns = [f"technique_{i}" for i in range(col_num)]

    _df = pd.concat([_df, pd.DataFrame(mat, columns=columns)], axis=1,)
    _df = _df.drop("name_", axis=1)

    raw.train = raw.train.merge(_df, on="object_id", how="left")
    raw.train.loc[:, columns] = raw.train.loc[:, columns].fillna(0).astype(np.int8)
    raw.test = raw.test.merge(_df, on="object_id", how="left")
    raw.test.loc[:, columns] = raw.test.loc[:, columns].fillna(0).astype(np.int8)
    return raw, col_num


def fe_cat_encoded_pca(
    raw: RawData, df: pd.DataFrame, col: str, n_components: int
) -> RawData:
    _df = df.copy()
    with open(Config.root_dir / "data/vecs.pickle", "rb") as f:
        vecs: Dict[str, np.ndarray] = pickle.load(f)
    train_encoded = np.stack(
        raw.train.merge(
            _df.groupby("object_id")["name"].apply(list).map(lambda l: ", ".join(l)),
            on="object_id",
            how="left",
        )["name"]
        .fillna("")
        .map(vecs)
        .values
    )
    test_encoded = np.stack(
        raw.test.merge(
            _df.groupby("object_id")["name"].apply(list).map(lambda l: ", ".join(l)),
            on="object_id",
            how="left",
        )["name"]
        .fillna("")
        .map(vecs)
        .values
    )
    concated = np.concatenate((train_encoded, test_encoded))
    with TimeUtil.timer(f"PCA {col}"):
        pca = PCA(n_components=n_components)
        pca.fit(concated)
        train_pca = pca.transform(train_encoded)
        test_pca = pca.transform(test_encoded)

    raw.train.loc[:, [f"{col}_pca_{i}" for i in range(n_components)]] = train_pca
    raw.test.loc[:, [f"{col}_pca_{i}" for i in range(n_components)]] = test_pca
    return raw


def fe_historical_person(raw: RawData) -> RawData:
    _counts = raw.historical_person["name"].value_counts()
    _dict = defaultdict(str)
    for row in tqdm(
        raw.historical_person.itertuples(), total=len(raw.historical_person)
    ):
        if (
            row.object_id not in _dict
            or _counts[_dict[row.object_id]] < _counts[row.name]
        ):
            _dict[row.object_id] = row.name
    raw.train["historical_person"] = raw.train["object_id"].map(_dict)
    raw.test["historical_person"] = raw.test["object_id"].map(_dict)
    return raw


def fe_object_collection(raw: RawData) -> RawData:
    _dict = defaultdict(list)
    _df = raw.object_collection.copy()
    _df["name"] = _df["name"].fillna("")
    _df.loc[~_df["name"].isin(["paintings", "prints", ""]), "name"] = "other"
    for row in tqdm(_df.itertuples(), total=len(_df)):
        _dict[row.object_id].append(row.name)
    raw.train["object_collection"] = raw.train["object_id"].map(_dict)
    raw.test["object_collection"] = raw.test["object_id"].map(_dict)
    for col in Config.object_collections:
        raw.train[f"object_collection_{col}"] = (
            raw.train["object_id"].map(lambda i: col in _dict[i]).astype(np.int8)
        )
        raw.test[f"object_collection_{col}"] = (
            raw.test["object_id"].map(lambda i: col in _dict[i]).astype(np.int8)
        )
        raw.train["object_collection_num"] = raw.train["object_id"].map(
            lambda i: len(_dict[i])
        )
        raw.test["object_collection_num"] = raw.test["object_id"].map(
            lambda i: len(_dict[i])
        )
    return raw


def fe_production_place(raw: RawData) -> RawData:
    _counts = raw.production_place["country"].value_counts()
    _dict = defaultdict(str)
    _num_dict = raw.production_place.groupby("object_id")["country"].count().to_dict()
    for row in tqdm(raw.production_place.itertuples(), total=len(raw.production_place)):
        if (
            row.object_id not in _dict
            or _counts[_dict[row.object_id]] < _counts[row.country]
        ):
            _dict[row.object_id] = row.country
    raw.train["country"] = raw.train["object_id"].map(_dict)
    raw.test["country"] = raw.test["object_id"].map(_dict)
    raw.train["country_num"] = raw.train["object_id"].map(_num_dict).fillna(0)
    raw.test["country_num"] = raw.test["object_id"].map(_num_dict).fillna(0)
    country_group_map = {
        "": "",
        "Netherlands": "Netherlands",
        "Belgium": "German",
        "Italy": "Latin",
        "France": "Latin",
        "Suriname": "Latin",
        "Germany": "German",
        "Indonesia": "Java",
        "Iran": "Java",
        "Norway": "German",
        "United Kingdom": "Latin",
        "United States": "America",
        "Denmark": "German",
        "Spain": "Latin",
        "Austria": "German",
        "Switzerland": "German",
        "Japan": "Asia",
        "India": "Asia",
        "China": "Asia",
        "Poland": "German",
        "Sri Lanka": "Java",
        "Greece": "Latin",
        "Canada": "America",
        "Russia": "German",
    }
    raw.train["country_group"] = raw.train["country"].map(country_group_map)
    raw.test["country_group"] = raw.test["country"].map(country_group_map)
    return raw


def fe_material(raw: RawData) -> RawData:
    _counts = raw.material["name"].value_counts()
    _dict = defaultdict(str)
    _num_dict = raw.material.groupby("object_id")["name"].count().to_dict()
    for row in tqdm(raw.material.itertuples(), total=len(raw.material)):
        if (
            row.object_id not in _dict
            or _counts[_dict[row.object_id]] < _counts[row.name]
        ):
            _dict[row.object_id] = row.name
    raw.train["material"] = raw.train["object_id"].map(_dict)
    raw.test["material"] = raw.test["object_id"].map(_dict)
    raw.train["material_num"] = raw.train["object_id"].map(_num_dict).fillna(0)
    raw.test["material_num"] = raw.test["object_id"].map(_num_dict).fillna(0)
    return raw


def fe_palette(raw: RawData) -> RawData:
    h = []
    s = []
    v = []
    for row in tqdm(raw.palette.itertuples(), total=len(raw.palette)):
        hsv = colorsys.rgb_to_hsv(row.color_r, row.color_g, row.color_b)
        h.append(hsv[0])
        s.append(hsv[1])
        v.append(hsv[2])
    raw.palette["color_h"] = h
    raw.palette["color_s"] = s
    raw.palette["color_v"] = v
    with TimeUtil.timer("palette sum"):
        palette = (
            raw.palette.groupby("object_id")
            .apply(
                lambda row: [
                    (row.ratio * row.color_r).sum(),
                    (row.ratio * row.color_g).sum(),
                    (row.ratio * row.color_b).sum(),
                ]
            )
            .to_dict()
        )
        hsv_palette = (
            raw.palette.groupby("object_id")
            .apply(
                lambda row: [
                    (row.ratio * row.color_h).sum(),
                    (row.ratio * row.color_s).sum(),
                    (row.ratio * row.color_v).sum(),
                ]
            )
            .to_dict()
        )
    raw.train["color_r"] = raw.train["object_id"].map(
        lambda i: palette[i][0] if i in palette else np.nan
    )
    raw.train["color_g"] = raw.train["object_id"].map(
        lambda i: palette[i][1] if i in palette else np.nan
    )
    raw.train["color_b"] = raw.train["object_id"].map(
        lambda i: palette[i][2] if i in palette else np.nan
    )
    raw.test["color_r"] = raw.test["object_id"].map(
        lambda i: palette[i][0] if i in palette else np.nan
    )
    raw.test["color_g"] = raw.test["object_id"].map(
        lambda i: palette[i][1] if i in palette else np.nan
    )
    raw.test["color_b"] = raw.test["object_id"].map(
        lambda i: palette[i][2] if i in palette else np.nan
    )

    raw.train["color_h"] = raw.train["object_id"].map(
        lambda i: hsv_palette[i][0] if i in hsv_palette else np.nan
    )
    raw.train["color_s"] = raw.train["object_id"].map(
        lambda i: hsv_palette[i][1] if i in hsv_palette else np.nan
    )
    raw.train["color_v"] = raw.train["object_id"].map(
        lambda i: hsv_palette[i][2] if i in hsv_palette else np.nan
    )
    raw.test["color_h"] = raw.test["object_id"].map(
        lambda i: hsv_palette[i][0] if i in hsv_palette else np.nan
    )
    raw.test["color_s"] = raw.test["object_id"].map(
        lambda i: hsv_palette[i][1] if i in hsv_palette else np.nan
    )
    raw.test["color_v"] = raw.test["object_id"].map(
        lambda i: hsv_palette[i][2] if i in hsv_palette else np.nan
    )
    for rgb in list("rgb"):
        _df = (
            raw.palette[raw.palette["ratio"] > 0.02]
            .groupby("object_id")[f"color_{rgb}"]
            .agg(["count", "mean", "std"])
        )
        _df.columns = [f"color_{rgb}_{agg}" for agg in ["count", "mean", "std"]]
        raw.train = raw.train.merge(_df, on="object_id", how="left")
        raw.test = raw.test.merge(_df, on="object_id", how="left")
        raw.train[f"color_{rgb}_count"] = raw.train[f"color_{rgb}_count"].fillna(0)
        raw.train[f"color_{rgb}_mean"] = raw.train[f"color_{rgb}_mean"].fillna(0)
        raw.test[f"color_{rgb}_count"] = raw.test[f"color_{rgb}_count"].fillna(0)
        raw.test[f"color_{rgb}_mean"] = raw.test[f"color_{rgb}_mean"].fillna(0)

    for hsv in list("hsv"):
        _df = (
            raw.palette[raw.palette["ratio"] > 0.02]
            .groupby("object_id")[f"color_{hsv}"]
            .agg(["mean", "std"])
        )
        _df.columns = [f"color_{hsv}_{agg}" for agg in ["mean", "std"]]
        raw.train = raw.train.merge(_df, on="object_id", how="left")
        raw.test = raw.test.merge(_df, on="object_id", how="left")
        raw.train[f"color_{hsv}_mean"] = raw.train[f"color_{hsv}_mean"].fillna(0)
        raw.test[f"color_{hsv}_mean"] = raw.test[f"color_{hsv}_mean"].fillna(0)
    return raw


def fe_color_pca(raw: RawData, n_components: int, pca_method: str):
    colors = defaultdict(list)
    ratios = defaultdict(list)
    _df = raw.palette.sort_values(["object_id", "ratio"], ascending=False)
    for row in tqdm(_df.itertuples(), total=len(_df)):
        for _ in range(6):
            ratios[row.object_id].append(row.ratio)
        colors[row.object_id].append(row.color_r)
        colors[row.object_id].append(row.color_g)
        colors[row.object_id].append(row.color_b)
        hsv = colorsys.rgb_to_hsv(row.color_r, row.color_g, row.color_b)
        colors[row.object_id].append(hsv[0])
        colors[row.object_id].append(hsv[1])
        colors[row.object_id].append(hsv[2])

    for object_id in (
        raw.train["object_id"].unique().tolist()
        + raw.test["object_id"].unique().tolist()
    ):
        if object_id not in colors:
            ratios[object_id] = [0] * 132
            colors[object_id] = [0] * 132

    train_encoded = np.stack(raw.train["object_id"].map(colors).values)
    test_encoded = np.stack(raw.test["object_id"].map(colors).values)

    n_components = 10
    pca_method = "PCA"

    concated = np.concatenate((train_encoded, test_encoded))
    with TimeUtil.timer(f"{pca_method} color"):
        if pca_method == "PCA":
            pca = PCA(n_components=n_components).fit(concated)
        elif pca_method == "TSVD":
            pca = TruncatedSVD(n_components=n_components).fit(concated)
        elif pca_method == "UMAP":
            pca = umap.UMAP(n_components=n_components).fit(concated)
        else:
            pca = PCA(n_components=n_components).fit(concated)
        train_pca = pca.transform(train_encoded)
        test_pca = pca.transform(test_encoded)
    raw.train.loc[:, [f"color_pca_{i}" for i in range(n_components)]] = train_pca
    raw.test.loc[:, [f"color_pca_{i}" for i in range(n_components)]] = test_pca
    return raw


def fe_encoded_pca(
    raw: RawData, n_components: int, col: str, with_tfidf: bool, pca_method: str
) -> RawData:
    train_encoded = np.load(
        Config.root_dir / f"data/train_{col}_encoded{Config.bert_model}.npy"
    )
    test_encoded = np.load(
        Config.root_dir / f"data/test_{col}_encoded{Config.bert_model}.npy"
    )

    if with_tfidf:
        stop_words = get_stop_words("dutch") + get_stop_words("en")
        tfidf = TfidfVectorizer(stop_words=stop_words, min_df=10)
        tfidf_vec = tfidf.fit(
            raw.train[col].fillna("").tolist() + raw.test[col].fillna("").tolist()
        )
        train_tfidf = tfidf_vec.transform(raw.train[col].fillna("")).todense()
        test_tfidf = tfidf_vec.transform(raw.test[col].fillna("")).todense()
        train_encoded = np.concatenate((train_encoded, train_tfidf), axis=1)
        test_encoded = np.concatenate((test_encoded, test_tfidf), axis=1)

    concated = np.concatenate((train_encoded, test_encoded))
    with TimeUtil.timer(f"{pca_method} {col}"):
        if pca_method == "PCA":
            pca = PCA(n_components=n_components).fit(concated)
        elif pca_method == "TSVD":
            pca = TruncatedSVD(n_components=n_components).fit(concated)
        elif pca_method == "UMAP":
            pca = umap.UMAP(n_components=n_components).fit(concated)
        else:
            pca = PCA(n_components=n_components).fit(concated)
        train_pca = pca.transform(train_encoded)
        test_pca = pca.transform(test_encoded)
    raw.train.loc[:, [f"{col}_pca_{i}" for i in range(n_components)]] = train_pca
    raw.test.loc[:, [f"{col}_pca_{i}" for i in range(n_components)]] = test_pca
    return raw


def fe_title_lang(raw: RawData) -> RawData:
    model = load_model(str(Config.root_dir / "data/lid.176.bin"))
    with TimeUtil.timer("title lang"):
        raw.train["title_lang"] = (
            raw.train["title"]
            .fillna("")
            .map(lambda x: model.predict(x.replace("\n", ""))[0][0])
        )
        raw.test["title_lang"] = (
            raw.test["title"]
            .fillna("")
            .map(lambda x: model.predict(x.replace("\n", ""))[0][0])
        )
    return raw


def target_encoding(
    train_df: pd.DataFrame, valid_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    target_mean = train_df["likes_log"].mean()
    _df = (
        train_df.groupby("principal_or_first_maker")["likes_log"]
        .agg("mean")
        .reset_index()
    )
    _df.columns = ["principal_or_first_maker", "principal_or_first_maker_target_mean"]
    train_df = train_df.merge(_df, on="principal_or_first_maker", how="left")
    valid_df = valid_df.merge(_df, on="principal_or_first_maker", how="left")
    train_df["principal_or_first_maker_target_mean"] = train_df[
        "principal_or_first_maker_target_mean"
    ].fillna(target_mean)
    valid_df["principal_or_first_maker_target_mean"] = valid_df[
        "principal_or_first_maker_target_mean"
    ].fillna(target_mean)
    return train_df, valid_df


def lgb_cv_tuning(
    dataset_x: pd.DataFrame,
    dataset_y: pd.DataFrame,
    cv_index: List[Tuple[pd.RangeIndex]],
) -> Tuple[lgb.Booster, dict]:
    train_dataset = lgb.Dataset(data=dataset_x, label=dataset_y)
    lgbtuner = LightGBMTunerCV(
        params={"objective": "regression", "metric": "rmse"},
        train_set=train_dataset,
        folds=cv_index,
        nfold=5,
        verbose_eval=False,
        num_boost_round=10000,
        early_stopping_rounds=100,
        return_cvbooster=True,
    )
    lgbtuner.run()
    models = lgbtuner.get_best_booster().boosters
    best_params = lgbtuner.best_params
    print("  Params: ")
    for key, value in best_params.items():
        print("    {}: {}".format(key, value))
    return models, best_params


def objective(trial: Trial):
    cat_train_dataset = Pool(
        _train_df[features], _train_df["likes_log"], cat_features=cat_features
    )
    cat_valid_dataset = Pool(
        _valid_df[features], _valid_df["likes_log"], cat_features=cat_features
    )
    params = {
        "depth": trial.suggest_int("depth", 4, 30),
        "num_leaves": trial.suggest_int("num_leaves", 16, 300),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 4, 50),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.001, 0.3),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 0.01, 0.5),
    }
    cat_model = CatBoostRegressor(**params, iterations=3500, grow_policy="Lossguide")
    cat_model.fit(
        cat_train_dataset,
        verbose_eval=100,
        eval_set=[cat_valid_dataset],
        early_stopping_rounds=200,
    )
    y_pred_cat = np.expm1(cat_model.predict(_valid_df[features]))
    y_pred_cat[y_pred_cat < 0] = 0
    y_true = _valid_df["likes"].values
    rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred_cat))
    return rmsle


def fe(raw: RawData,) -> RawData:
    raw = fe_sub_title(raw)
    raw = fe_dating(raw)
    raw = fe_historical_person(raw)
    raw = fe_material(raw)
    raw = fe_colorspace_pca(raw)
    raw, Config.technique_col_num = fe_technique(raw)
    raw = fe_color_pca(raw, Config.color_n_components, "PCA")
    raw = fe_cat_encoded_pca(
        raw, raw.material, "material", Config.material_n_components
    )
    raw = fe_cat_encoded_pca(
        raw, raw.technique, "technique", Config.material_n_components
    )
    raw = fe_encoded_pca(
        raw,
        Config.title_maker_n_components,
        "title_principal_or_first_maker",
        True,
        "PCA",
    )
    raw = fe_encoded_pca(raw, Config.title_n_components, "title", True, "PCA")
    raw = fe_encoded_pca(
        raw, Config.long_title_n_components, "long_title", False, "PCA"
    )
    # raw = fe_encoded_pca(raw, Config.desc_n_components, "description", True, "PCA")
    raw = fe_encoded_pca(
        raw, Config.desc_en_n_components, "description_en", True, "UMAP"
    )
    raw = fe_encoded_pca(
        raw, Config.principal_maker_n_components, "principal_maker", False, "PCA"
    )
    # raw = fe_encoded_pca(
    #     raw,
    #     Config.long_title_desc_en_n_components,
    #     "long_title_description_en",
    #     True,
    #     "PCA",
    # )
    raw = fe_title_lang(raw)
    raw = fe_color(raw)
    raw = fe_object_collection(raw)
    raw = fe_production_place(raw)
    raw = fe_palette(raw)
    return raw


def fillna(raw: RawData) -> RawData:
    fill_zero_cols = [
        "color_r",
        "color_g",
        "color_b",
        "color_h",
        "color_s",
        "color_v",
        "sub_title_len",
    ]
    fill_mean_cols = [
        "size_h",
        "size_w",
        "size_t",
        "size_d",
        "size_hw_ratio",
        "size_area",
        "dating_year_early",
        "dating_year_late",
        "color_r_std",
        "color_g_std",
        "color_b_std",
        "color_h_std",
        "color_s_std",
        "color_v_std",
    ] + [f"colorspace_pca_{i}" for i in range(Config.colorspace_n_components)]
    _df = pd.concat([raw.train, raw.test], axis=0).reset_index(drop=True)
    for col in fill_zero_cols:
        raw.train[col] = raw.train[col].fillna(0)
        raw.test[col] = raw.test[col].fillna(0)

    for col in fill_mean_cols:
        raw.train[col] = raw.train[col].fillna(_df[col].mean())
        raw.test[col] = raw.test[col].fillna(_df[col].mean())
    return raw


def get_pseudo_df(raw: RawData) -> pd.DataFrame:
    raw.train["is_train"] = 1
    raw.test["is_train"] = 0
    _df = pd.concat([raw.train, raw.test], axis=0)
    group = _df.groupby(["principal_maker", "size_h", "size_w"])
    pseudo_df = pd.DataFrame()
    for (principal_maker, size_h, size_w), df in tqdm(group, total=len(group)):
        _train_df = df[df["is_train"] == 1].reset_index(drop=True)
        _test_df = df[df["is_train"] == 0].reset_index(drop=True)
        train_len = len(_train_df)
        test_len = len(_test_df)
        if train_len > 1 and test_len > 0:
            likes_log_mean = _train_df["likes_log"].mean()
            _test_df["likes_log"] = likes_log_mean
            pseudo_df = pd.concat([pseudo_df, _test_df], axis=0)
    return pseudo_df


# %%

with TimeUtil.timer("load raw data"):
    raw = RawData(
        train=pd.read_csv(Config.data_dir / "train.csv"),
        test=pd.read_csv(Config.data_dir / "test.csv"),
        color=pd.read_csv(Config.data_dir / "color.csv"),
        historical_person=pd.read_csv(Config.data_dir / "historical_person.csv"),
        maker=pd.read_csv(Config.data_dir / "maker.csv"),
        material=pd.read_csv(Config.data_dir / "material.csv"),
        object_collection=pd.read_csv(Config.data_dir / "object_collection.csv"),
        palette=pd.read_csv(Config.data_dir / "palette.csv"),
        principal_maker_occupation=pd.read_csv(
            Config.data_dir / "principal_maker_occupation.csv"
        ),
        principal_maker=pd.read_csv(Config.data_dir / "principal_maker.csv"),
        production_place=pd.read_csv(Config.data_dir / "production_place.csv"),
        technique=pd.read_csv(Config.data_dir / "technique.csv"),
        colorspace_selected_pca=pd.read_csv(
            Config.data_dir / "colorspace_selected_pca.csv"
        ),
        sample_submission=pd.read_csv(
            Config.data_dir / "atmacup10__sample_submission.csv"
        ),
    )

GlobalUtil.seed_everything(Config.seed)
raw.train["likes_log"] = np.log1p(raw.train["likes"])
raw.train["long_title_description_en"] = (
    raw.train["long_title"] + " " + raw.train["description_en"]
)
raw.train["title_principal_or_first_maker"] = (
    raw.train["title"] + " " + raw.train["principal_or_first_maker"]
)
raw.test["title_principal_or_first_maker"] = (
    raw.test["title"] + " " + raw.test["principal_or_first_maker"]
)
raw = fe(raw)


# %%

models = "lgbm"
features = (
    [
        "size_h",
        "size_w",
        "size_t",
        "size_d",
        "size_area",
        "size_hw_ratio",
        "max_percentage",
        # "max_hex",
        "principal_maker",
        "principal_or_first_maker",
        "copyright_holder",
        "acquisition_method",
        "acquisition_credit_line",
        "dating_period",
        "dating_year_early",
        "dating_year_late",
        "historical_person",
        "country_group",
        "country_num",
        # "material",
        # "technique",
        # "object_collection",
        # "production_place",
        # "color_r",
        # "color_g",
        # "color_b",
        # "color_h",
        # "color_s",
        # "color_v",
        "title_lang",
        "title_len",
        "title_word_num",
        "title_capital_word_num",
        "sub_title_len",
        "title_contains_the",
        "color_count",
        "material_num",
        # "description_en_word_num",
        # "description_en_len",
        # "dating_acquisition_diff",
        # "dating_year_diff",
        # "principal_or_first_maker_target_mean"
        "object_collection_num",
        "more_title_is_less_than_title",
    ]
    + [f"color_{rgb}_{agg}" for rgb in list("rgb") for agg in ["count", "mean", "std"]]
    + [f"color_{hsv}_{agg}" for hsv in list("hsv") for agg in ["mean", "std"]]
    # + [
    #     f"title_principal_or_first_maker_pca_{i}"
    #     for i in range(Config.title_n_components)
    # ]
    + [f"title_pca_{i}" for i in range(Config.title_n_components)]
    + [f"long_title_pca_{i}" for i in range(Config.long_title_n_components)]
    + [f"description_en_pca_{i}" for i in range(Config.desc_en_n_components)]
    # + [f"description_pca_{i}" for i in range(Config.desc_n_components)]
    + [f"principal_maker_pca_{i}" for i in range(Config.principal_maker_n_components)]
    # + [f"technique_{i}" for i in range(Config.technique_col_num)]
    # + [
    #     f"long_title_description_en_pca_{i}"
    #     for i in range(Config.long_title_desc_en_n_components)
    # ]
    # + [f"color_pca_{i}" for i in range(Config.color_n_components)]
    # + [f"colorspace_pca_{i}" for i in range(Config.colorspace_n_components)]
    + [f"material_pca_{i}" for i in range(Config.material_n_components)]
    + [f"technique_pca_{i}" for i in range(Config.technique_n_components)]
    + [f"object_collection_{col}" for col in Config.object_collections]
)

cat_features = [
    f
    for f in features
    if f
    in [
        "principal_maker",
        "principal_or_first_maker",
        "copyright_holder",
        "acquisition_method",
        "acquisition_credit_line",
        "max_hex",
        "technique",
        "historical_person",
        "object_collection",
        "production_place",
        "material",
        "title_lang",
        "country_group",
        "country",
    ]
]

features += [f"{col}_{agg}" for agg in Config.cat_aggs for col in cat_features]


label_encs = {col: LabelEncoder() for col in cat_features}
for col in cat_features:
    label_encs, raw = label_encoding(label_encs, raw, col)
    raw = cat_encoding(raw, col)

raw = fillna(raw)
folds = StratifiedKFold(
    n_splits=Config.n_splits, shuffle=True, random_state=Config.seed
).split(raw.train["object_id"], raw.train["likes"])

if models == "lgbo":
    _folds = list(folds)
    lgb_cv_tuning(raw.train[features], raw.train["likes_log"], _folds)
    os._exit(0)

MlflowUtil.start_run(models)
MlflowUtil.log_config()
mlflow.log_param("model", models)
mlflow.log_param("features", "\n,".join(features))
mlflow.log_param("cat_features", "\n,".join(cat_features))
if "lgbm" in models:
    mlflow.lightgbm.autolog()
rmsles = []
lgb_models = {}
cat_models = {}
xgb_models = {}
studies = {}
# train_folds = [3, 4]
train_folds = [0, 1, 2, 3, 4]
for fold, (train_idx, valid_idx) in enumerate(folds):
    if fold not in train_folds:
        continue
    print(f"------------------------ fold {fold} -----------------------")
    _train_df = raw.train.loc[train_idx]
    _valid_df = raw.train.loc[valid_idx]

    if models == "cato":
        study = optuna.create_study()
        study.optimize(objective, n_trials=20)
        studies[fold] = study

    # _train_df, _valid_df = target_encoding(_train_df, _valid_df)

    if "lgbm" in models:
        lgb_train_dataset = lgb.Dataset(_train_df[features], _train_df["likes_log"])
        lgb_valid_dataset = lgb.Dataset(_valid_df[features], _valid_df["likes_log"])
        lgb_model = lgb.train(
            Config.lgb_params,
            lgb_train_dataset,
            num_boost_round=2000,
            valid_sets=[lgb_train_dataset, lgb_valid_dataset],
            verbose_eval=50,
            early_stopping_rounds=300,
            categorical_feature=cat_features,
        )
        y_pred_lgb = np.expm1(lgb_model.predict(_valid_df[features]))
        lgb_models[fold] = lgb_model

    if "cat" in models:
        cat_train_dataset = Pool(
            _train_df[features], _train_df["likes_log"], cat_features=cat_features
        )
        cat_valid_dataset = Pool(
            _valid_df[features], _valid_df["likes_log"], cat_features=cat_features
        )
        cat_model = CatBoostRegressor(**Config.cat_params, iterations=4500)
        cat_model.fit(
            cat_train_dataset,
            verbose_eval=100,
            eval_set=[cat_valid_dataset],
            early_stopping_rounds=200,
        )
        y_pred_cat = np.expm1(cat_model.predict(_valid_df[features]))
        cat_models[fold] = cat_model

    if "xgb" in models:
        xgb_train_dataset = xgb.DMatrix(
            _train_df[features].values, label=_train_df["likes_log"].values
        )
        xgb_valid_dataset = xgb.DMatrix(
            _valid_df[features].values, label=_valid_df["likes_log"].values
        )
        evals = [(xgb_train_dataset, "train"), (xgb_valid_dataset, "eval")]
        evals_result = {}
        xgb_model = xgb.train(
            Config.xgb_params,
            xgb_train_dataset,
            evals=evals,
            evals_result=evals_result,
            num_boost_round=2000,
            early_stopping_rounds=100,
            verbose_eval=50,
        )
        y_pred_xgb = np.expm1(xgb_model.predict(xgb_valid_dataset))
        xgb_models[fold] = xgb_model

    if models == "lgbm+cat":
        y_pred = 0.5 * y_pred_cat + 0.5 * y_pred_lgb
    elif models == "lgbm+cat+xgb":
        y_pred = 0.4 * y_pred_cat + 0.4 * y_pred_lgb + 0.2 * y_pred_xgb
    elif models == "lgbm":
        y_pred = y_pred_lgb
    elif models == "cat":
        y_pred = y_pred_cat
    elif models == "xgb":
        y_pred = y_pred_xgb

    if models != "cato":
        y_pred[y_pred < 0] = 0
        y_true = _valid_df["likes"].values
        rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))
        rmsles.append(rmsle)
        mlflow.log_metric(f"rmsle_{fold}", rmsle)
        print(f"------------------------ fold {fold} -----------------------")
        print(f"------------------- rmsle {rmsle} -----------------------")
        print()

print("")
print(f"------------------- average rmsle {np.mean(rmsles)} -----------------------")
mlflow.log_metric(f"rmsle_avg", np.mean(rmsles))
mlflow.end_run()
if "lgbm" in models:
    lgb.plot_importance(lgb_model, importance_type="split", figsize=(16, 24))
    plt.title("split")
    plt.show()

    lgb.plot_importance(lgb_model, importance_type="gain", figsize=(16, 24))
    plt.title("gain")
    plt.show()

# %%
# pseudo_df = get_pseudo_df(raw)
# raw.train = pd.concat([raw.train, pseudo_df], axis=0).reset_index(drop=True)
# raw.train, raw.test = target_encoding(raw.train, raw.test)
cat_train_dataset = Pool(
    raw.train[features], raw.train["likes_log"], cat_features=cat_features
)
lgb_train_dataset = lgb.Dataset(raw.train[features], raw.train["likes_log"])
cat_model = CatBoostRegressor(**Config.cat_params, iterations=4000)
cat_model.fit(
    cat_train_dataset, verbose_eval=100, eval_set=[cat_train_dataset],
)
lgb_model = lgb.train(
    Config.lgb_params,
    lgb_train_dataset,
    num_boost_round=1300,
    verbose_eval=50,
    valid_sets=[lgb_train_dataset],
    categorical_feature=cat_features,
)
test_pred_cat = np.expm1(cat_model.predict(raw.test[features]))
test_pred_lgb = np.expm1(lgb_model.predict(raw.test[features]))
test_pred = 0.5 * test_pred_cat + 0.5 * test_pred_lgb
test_pred[test_pred < 0] = 0
raw.sample_submission["likes"] = test_pred
raw.sample_submission.to_csv(Path.cwd() / "output" / "exp016_3.csv", index=False)

# %%
raw.test = raw.test.merge(
    pseudo_df[["object_id", "likes_log"]], on="object_id", how="left"
)
assert len(raw.sample_submission) == len(raw.test)
raw.sample_submission["likes_log"] = raw.test["likes_log"]
raw.sample_submission["likes"] = raw.sample_submission.apply(
    lambda row: np.expm1(row.likes_log)
    if row.likes_log == row.likes_log
    else row.likes,
    axis=1,
)
raw.sample_submission.drop(["likes_log"], axis=1, inplace=True)
raw.sample_submission.to_csv(Path.cwd() / "output" / "exp014-2_1.csv", index=False)


# %%
test_pred_all = np.zeros((10, len(raw.test)))
for fold in range(Config.n_splits):
    test_pred_all[fold, :] = np.expm1(cat_models[fold].predict(raw.test[features]))
for fold in range(Config.n_splits):
    test_pred_all[5 + fold, :] = np.expm1(lgb_models[fold].predict(raw.test[features]))
test_pred = test_pred_all.mean(axis=0)
test_pred[test_pred < 0] = 0
raw.sample_submission["likes"] = test_pred
raw.sample_submission.to_csv(Path.cwd() / "output" / "exp014-1_1.csv", index=False)

# %%
