# %%
import colorsys
import math
import os
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
import pandas as pd
import psutil
import pycld2 as cld2
import seaborn as sns
import torch
import torch.backends
import torch.backends.cudnn
import torch.cuda
import umap
from catboost import CatBoostRegressor, Pool
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from stop_words import get_stop_words
from tqdm.autonotebook import tqdm


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
    cat_aggs = ["count"]
    # bert_model = "_stsb_roberta_large"
    # bert_model = "_bert_base_multilingual_uncase"
    # bert_model = "_bert_base_multilingual_case"
    bert_model = "_bert_base_multilingual_case_mean"
    # bert_model = ""
    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "feature_pre_filter": False,
        "lambda_l1": 0.0006114213973711655,
        "lambda_l2": 0.0012614682742735897,
        "num_leaves": 89,
        "feature_fraction": 0.584,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "min_child_samples": 20,
    }
    lgb_params = {
        "num_leaves": 89,
        "min_data_in_leaf": 20,
        "objective": "regression",
        "max_depth": -1,
        "learning_rate": 0.03,
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
        "seed": seed,
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
        sample_submission=pd.read_csv(
            Config.data_dir / "atmacup10__sample_submission.csv"
        ),
    )

raw.train["likes_log"] = np.log1p(raw.train["likes"])
raw.train["long_title_description_en"] = (
    raw.train["long_title"] + " " + raw.train["description_en"]
)

# %%


def fe_technique(raw: RawData) -> RawData:
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
    return raw


def fe_object_collection(raw: RawData) -> RawData:
    _counts = raw.object_collection["name"].value_counts()
    _dict = defaultdict(str)
    raw.object_collection["name"] = raw.object_collection["name"].fillna("")
    raw.object_collection.loc[
        ~raw.object_collection["name"].isin(["paintings", "prints", ""]), "name"
    ] = "other"
    for row in tqdm(
        raw.object_collection.itertuples(), total=len(raw.object_collection)
    ):
        if (
            row.object_id not in _dict
            or _counts[_dict[row.object_id]] < _counts[row.name]
        ):
            _dict[row.object_id] = row.name
    raw.train["object_collection"] = raw.train["object_id"].map(_dict)
    raw.test["object_collection"] = raw.test["object_id"].map(_dict)
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
    return df


def fe_sub_title(raw: RawData) -> RawData:
    raw.train = _fe_sub_title(raw.train)
    raw.test = _fe_sub_title(raw.test)
    return raw


# raw = fe_technique(raw)
raw = fe_object_collection(raw)
# raw = fe_material(raw)
# raw.train["more_title_is_less_than_title"] = raw.train.apply(
#     lambda row: len(row.more_title) < len(row.title), axis=1
# ).astype(np.int8)
# raw.test["more_title_is_less_than_title"] = raw.test.apply(
#     lambda row: len(row.more_title) < len(row.title), axis=1
# ).astype(np.int8)

raw = fe_production_place(raw)
raw = fe_sub_title(raw)

featrues = [
    "size_hw_ratio",
    "country_group",
    "material_num",
]

# technique -> 数と単語ベース
# material -> 数と単語ベース

# %%
raw.train["size_hw_ratio"] = raw.train["size_h"] / raw.train["size_w"]
raw.test["size_hw_ratio"] = raw.test["size_h"] / raw.test["size_w"]

# %%
dfs: List[pd.DataFrame] = []
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
    if train_len > 2 and test_len > 0:
        likes_log_mean = _train_df["likes_log"].mean()
        _test_df["likes_log"] = likes_log_mean
        pseudo_df = pd.concat([pseudo_df, _test_df], axis=0)

# %%

# %%
dfs[20][
    [
        "title",
        "principal_maker",
        "principal_or_first_maker",
        "object_collection",
        "likes",
        "likes_log",
    ]
]


# %%
raw.sample_submission.loc[8585, "likes"] = 29.2
raw.sample_submission.loc[10125, "likes"] = 29.2


# %%
raw.train.groupby("country_group")["likes_log"].agg(["count", "mean"])


# %%
raw.train.groupby("object_collection")["likes_log"].agg(["count", "mean"]).sort_values(
    "mean", ascending=False
)


# %%
# raw.train.query("technique == ''")[["title", "likes_log"]].sort_values("likes_log", ascending=False)[:20]
raw.train[["title", "technique", "likes_log"]].sort_values("likes_log", ascending=True)[
    60:80
]

# %%
raw.train.query("technique == 'cyanotype'")[["title", "likes_log"]].sort_values(
    "likes_log", ascending=False
)
