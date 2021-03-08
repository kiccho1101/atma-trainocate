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
from typing import Any, Dict, Optional, Tuple

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
    cat_aggs = ["count"]
    # bert_model = "_stsb_roberta_large"
    # bert_model = "_bert_base_multilingual_uncase"
    bert_model = "_bert_base_multilingual_case"
    # bert_model = ""
    lgb_params = {
        "num_leaves": 64,
        "min_data_in_leaf": 32,
        "objective": "regression",
        "max_depth": -1,
        "learning_rate": 0.05,
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
    cat_params = {
        "min_data_in_leaf": 16,
        "learning_rate": 0.05,
        "reg_lambda": 0.3,
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
    return df


def fe_sub_title(raw: RawData) -> RawData:
    raw.train = _fe_sub_title(raw.train)
    raw.test = _fe_sub_title(raw.test)
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
    _counts = raw.object_collection["name"].value_counts()
    _dict = defaultdict(str)
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


def fe_production_place(raw: RawData) -> RawData:
    _counts = raw.production_place["name"].value_counts()
    _dict = defaultdict(str)
    for row in tqdm(raw.production_place.itertuples(), total=len(raw.production_place)):
        if (
            row.object_id not in _dict
            or _counts[_dict[row.object_id]] < _counts[row.name]
        ):
            _dict[row.object_id] = row.name
    raw.train["production_place"] = raw.train["object_id"].map(_dict)
    raw.test["production_place"] = raw.test["object_id"].map(_dict)
    return raw


def fe_material(raw: RawData) -> RawData:
    _counts = raw.material["name"].value_counts()
    _dict = defaultdict(str)
    for row in tqdm(raw.material.itertuples(), total=len(raw.material)):
        if (
            row.object_id not in _dict
            or _counts[_dict[row.object_id]] < _counts[row.name]
        ):
            _dict[row.object_id] = row.name
    raw.train["material"] = raw.train["object_id"].map(_dict)
    raw.test["material"] = raw.test["object_id"].map(_dict)
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
            .agg(["count", "mean", "var"])
        )
        _df.columns = [f"color_{rgb}_{agg}" for agg in ["count", "mean", "var"]]
        raw.train = raw.train.merge(_df, on="object_id", how="left")
        raw.test = raw.test.merge(_df, on="object_id", how="left")
        raw.train[f"color_{rgb}_count"] = raw.train[f"color_{rgb}_count"].fillna(0)
        raw.train[f"color_{rgb}_mean"] = raw.train[f"color_{rgb}_mean"].fillna(0)

    for hsv in list("hsv"):
        _df = (
            raw.palette[raw.palette["ratio"] > 0.02]
            .groupby("object_id")[f"color_{hsv}"]
            .agg(["mean", "var"])
        )
        _df.columns = [f"color_{hsv}_{agg}" for agg in ["mean", "var"]]
        raw.train = raw.train.merge(_df, on="object_id", how="left")
        raw.test = raw.test.merge(_df, on="object_id", how="left")
        raw.train[f"color_{hsv}_mean"] = raw.train[f"color_{hsv}_mean"].fillna(0)
        raw.test[f"color_{hsv}_mean"] = raw.test[f"color_{hsv}_mean"].fillna(0)
    return raw


def fe_title_pca(raw: RawData, n_components: int) -> RawData:
    col = "title"

    stop_words = get_stop_words("dutch") + get_stop_words("en")
    tfidf = TfidfVectorizer(stop_words=stop_words, min_df=10)
    tfidf_vec = tfidf.fit(
        raw.train[col].fillna("").tolist() + raw.test[col].fillna("").tolist()
    )
    train_tfidf = tfidf_vec.transform(raw.train[col].fillna("")).todense()
    test_tfidf = tfidf_vec.transform(raw.test[col].fillna("")).todense()

    train_encoded = np.load(
        Config.root_dir / f"data/train_{col}_encoded{Config.bert_model}.npy"
    )
    test_encoded = np.load(
        Config.root_dir / f"data/test_{col}_encoded{Config.bert_model}.npy"
    )

    train_encoded = np.concatenate((train_encoded, train_tfidf), axis=1)
    test_encoded = np.concatenate((test_encoded, test_tfidf), axis=1)

    concated = np.concatenate((train_encoded, test_encoded))

    with TimeUtil.timer("pca title"):
        if Config.pca_method == "PCA":
            pca = PCA(n_components=n_components).fit(concated)
        elif Config.pca_method == "TSVD":
            pca = TruncatedSVD(n_components=n_components).fit(concated)
        elif Config.pca_method == "TSNE":
            pca = TSNE(n_components=n_components).fit(concated)
        elif Config.pca_method == "UMAP":
            pca = umap.UMAP(n_components=n_components).fit(concated)
        else:
            pca = PCA(n_components=n_components).fit(concated)
    train_pca = pca.transform(train_encoded)
    test_pca = pca.transform(test_encoded)
    raw.train.loc[:, [f"{col}_pca_{i}" for i in range(n_components)]] = train_pca
    raw.test.loc[:, [f"{col}_pca_{i}" for i in range(n_components)]] = test_pca
    return raw


def fe_long_title_pca(raw: RawData, n_components: int) -> RawData:
    col = "long_title"

    stop_words = get_stop_words("dutch") + get_stop_words("en")
    tfidf = TfidfVectorizer(stop_words=stop_words, min_df=10)
    tfidf_vec = tfidf.fit(
        raw.train[col].fillna("").tolist() + raw.test[col].fillna("").tolist()
    )
    train_tfidf = tfidf_vec.transform(raw.train[col].fillna("")).todense()
    test_tfidf = tfidf_vec.transform(raw.test[col].fillna("")).todense()

    train_encoded = np.load(
        Config.root_dir / f"data/train_{col}_encoded{Config.bert_model}.npy"
    )
    test_encoded = np.load(
        Config.root_dir / f"data/test_{col}_encoded{Config.bert_model}.npy"
    )

    train_encoded = np.concatenate((train_encoded, train_tfidf), axis=1)
    test_encoded = np.concatenate((test_encoded, test_tfidf), axis=1)

    concated = np.concatenate((train_encoded, test_encoded))
    with TimeUtil.timer("pca long_title"):
        if Config.pca_method == "PCA":
            pca = PCA(n_components=n_components).fit(concated)
        elif Config.pca_method == "TSVD":
            pca = TruncatedSVD(n_components=n_components).fit(concated)
        elif Config.pca_method == "UMAP":
            pca = umap.UMAP(n_components=n_components).fit(concated)
        else:
            pca = PCA(n_components=n_components).fit(concated)
    train_pca = pca.transform(train_encoded)
    test_pca = pca.transform(test_encoded)
    raw.train.loc[:, [f"{col}_pca_{i}" for i in range(n_components)]] = train_pca
    raw.test.loc[:, [f"{col}_pca_{i}" for i in range(n_components)]] = test_pca
    return raw


def fe_description_en_pca(raw: RawData, n_components: int) -> RawData:
    col = "description_en"
    train_encoded = np.load(
        Config.root_dir / f"data/train_{col}_encoded{Config.bert_model}.npy"
    )
    test_encoded = np.load(
        Config.root_dir / f"data/test_{col}_encoded{Config.bert_model}.npy"
    )
    concated = np.concatenate((train_encoded, test_encoded))
    with TimeUtil.timer("pca description_en"):
        if Config.pca_method == "PCA":
            pca = PCA(n_components=n_components).fit(concated)
        elif Config.pca_method == "TSVD":
            pca = TruncatedSVD(n_components=n_components).fit(concated)
        elif Config.pca_method == "UMAP":
            pca = umap.UMAP(n_components=n_components).fit(concated)
        else:
            pca = PCA(n_components=n_components).fit(concated)
    train_pca = pca.transform(train_encoded)
    test_pca = pca.transform(test_encoded)
    raw.train.loc[:, [f"{col}_pca_{i}" for i in range(n_components)]] = train_pca
    raw.test.loc[:, [f"{col}_pca_{i}" for i in range(n_components)]] = test_pca
    return raw


def fe_principal_maker_pca(raw: RawData, n_components: int) -> RawData:
    col = "principal_maker"
    train_encoded = np.load(
        Config.root_dir / f"data/train_{col}_encoded{Config.bert_model}.npy"
    )
    test_encoded = np.load(
        Config.root_dir / f"data/test_{col}_encoded{Config.bert_model}.npy"
    )
    concated = np.concatenate((train_encoded, test_encoded))
    with TimeUtil.timer("pca principal_maker"):
        if Config.pca_method == "PCA":
            pca = PCA(n_components=n_components).fit(concated)
        elif Config.pca_method == "TSVD":
            pca = TruncatedSVD(n_components=n_components).fit(concated)
        elif Config.pca_method == "UMAP":
            pca = umap.UMAP(n_components=n_components).fit(concated)
        else:
            pca = PCA(n_components=n_components).fit(concated)
    train_pca = pca.transform(train_encoded)
    test_pca = pca.transform(test_encoded)
    raw.train.loc[:, [f"{col}_pca_{i}" for i in range(n_components)]] = train_pca
    raw.test.loc[:, [f"{col}_pca_{i}" for i in range(n_components)]] = test_pca
    return raw


def fe_description_pca(raw: RawData, n_components: int) -> RawData:
    col = "description"

    stop_words = get_stop_words("dutch") + get_stop_words("en")
    tfidf = TfidfVectorizer(stop_words=stop_words, min_df=10)
    tfidf_vec = tfidf.fit(
        raw.train[col].fillna("").tolist() + raw.test[col].fillna("").tolist()
    )
    train_tfidf = tfidf_vec.transform(raw.train[col].fillna("")).todense()
    test_tfidf = tfidf_vec.transform(raw.test[col].fillna("")).todense()

    train_encoded = np.load(
        Config.root_dir / f"data/train_{col}_encoded{Config.bert_model}.npy"
    )
    test_encoded = np.load(
        Config.root_dir / f"data/test_{col}_encoded{Config.bert_model}.npy"
    )

    train_encoded = np.concatenate((train_encoded, train_tfidf), axis=1)
    test_encoded = np.concatenate((test_encoded, test_tfidf), axis=1)

    concated = np.concatenate((train_encoded, test_encoded))
    with TimeUtil.timer("pca description"):
        if Config.pca_method == "PCA":
            pca = PCA(n_components=n_components).fit(concated)
        elif Config.pca_method == "TSVD":
            pca = TruncatedSVD(n_components=n_components).fit(concated)
        elif Config.pca_method == "UMAP":
            pca = umap.UMAP(n_components=n_components).fit(concated)
        else:
            pca = PCA(n_components=n_components).fit(concated)
    train_pca = pca.transform(train_encoded)
    test_pca = pca.transform(test_encoded)
    raw.train.loc[:, [f"{col}_pca_{i}" for i in range(n_components)]] = train_pca
    raw.test.loc[:, [f"{col}_pca_{i}" for i in range(n_components)]] = test_pca
    return raw


def fe_long_title_description_en_pca(raw: RawData, n_components: int) -> RawData:
    col = "long_title_description_en"

    train_encoded = np.load(
        Config.root_dir / f"data/train_{col}_encoded{Config.bert_model}.npy"
    )
    test_encoded = np.load(
        Config.root_dir / f"data/test_{col}_encoded{Config.bert_model}.npy"
    )
    concated = np.concatenate((train_encoded, test_encoded))
    if Config.pca_method == "PCA":
        pca = PCA(n_components=n_components).fit(concated)
    elif Config.pca_method == "TSVD":
        pca = TruncatedSVD(n_components=n_components).fit(concated)
    elif Config.pca_method == "UMAP":
        pca = umap.UMAP(n_components=n_components).fit(concated)
    else:
        pca = PCA(n_components=n_components).fit(concated)
    train_pca = pca.transform(train_encoded)
    test_pca = pca.transform(test_encoded)
    raw.train.loc[:, [f"{col}_pca_{i}" for i in range(n_components)]] = train_pca
    raw.test.loc[:, [f"{col}_pca_{i}" for i in range(n_components)]] = test_pca
    return raw


def fe_title_lang(raw: RawData) -> RawData:
    raw.train["title_lang"] = (
        raw.train["title"].fillna("").map(lambda x: cld2.detect(x)[2][0][1])
    )
    raw.test["title_lang"] = (
        raw.test["title"].fillna("").map(lambda x: cld2.detect(x)[2][0][1])
    )
    return raw


def fe(
    raw: RawData,
    title_n_components: int,
    long_title_n_components: int,
    desc_en_n_components: int,
    desc_n_components: int,
    principal_maker_n_components: int,
    long_title_desc_en_n_components: int,
) -> RawData:
    raw = fe_sub_title(raw)
    raw = fe_historical_person(raw)
    raw = fe_material(raw)
    raw = fe_title_pca(raw, title_n_components)
    raw = fe_long_title_pca(raw, long_title_n_components)
    raw = fe_description_en_pca(raw, desc_en_n_components)
    raw = fe_principal_maker_pca(raw, principal_maker_n_components)
    # raw = fe_long_title_description_en_pca(raw, long_title_desc_en_n_components)
    raw = fe_description_pca(raw, desc_n_components)
    raw = fe_title_lang(raw)
    raw = fe_color(raw)
    raw = fe_object_collection(raw)
    raw = fe_technique(raw)
    raw = fe_production_place(raw)
    raw = fe_palette(raw)
    return raw


def fillna(raw: RawData) -> RawData:
    fill_zero_cols = [
        "size_h",
        "size_w",
        "size_t",
        "size_d",
        "size_area",
        "color_r",
        "color_g",
        "color_b",
        "sub_title_len",
        "color_r_var",
        "color_g_var",
        "color_b_var",
    ]
    fill_mean_cols = [
        "dating_year_early",
        "dating_year_late",
    ]
    for col in fill_zero_cols:
        raw.train[col] = raw.train[col].fillna(0)
        raw.test[col] = raw.test[col].fillna(0)

    for col in fill_mean_cols:
        raw.train[col] = raw.train[col].fillna(raw.train[col].mean())
        raw.test[col] = raw.test[col].fillna(raw.train[col].mean())
    return raw


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
        sample_submission=pd.read_csv(
            Config.data_dir / "atmacup10__sample_submission.csv"
        ),
    )

GlobalUtil.seed_everything(Config.seed)
raw.train["likes_log"] = np.log1p(raw.train["likes"])
raw = fe(
    raw,
    Config.title_n_components,
    Config.long_title_n_components,
    Config.desc_en_n_components,
    Config.desc_n_components,
    Config.principal_maker_n_components,
    Config.long_title_desc_en_n_components,
)


# %%
models = "lgbm"
features = (
    [
        "size_h",
        "size_w",
        "size_t",
        "size_d",
        "size_area",
        "max_percentage",
        "max_hex",
        "principal_maker",
        "principal_or_first_maker",
        "copyright_holder",
        "acquisition_method",
        "acquisition_credit_line",
        "dating_period",
        "dating_year_early",
        "dating_year_late",
        "technique",
        "historical_person",
        "object_collection",
        "production_place",
        "material",
        "color_r",
        "color_g",
        "color_b",
        "color_h",
        "color_s",
        "color_v",
        "title_lang",
        "title_len",
        "sub_title_len",
        "color_count",
    ]
    + [f"color_{rgb}_{agg}" for rgb in list("rgb") for agg in ["count", "mean", "var"]]
    + [f"color_{hsv}_{agg}" for hsv in list("hsv") for agg in ["mean", "var"]]
    + [f"title_pca_{i}" for i in range(Config.title_n_components)]
    + [f"long_title_pca_{i}" for i in range(Config.long_title_n_components)]
    + [f"description_en_pca_{i}" for i in range(Config.desc_en_n_components)]
    # + [f"description_pca_{i}" for i in range(Config.desc_n_components)]
    + [f"principal_maker_pca_{i}" for i in range(Config.principal_maker_n_components)]
    # + [
    #     f"long_title_description_en_pca_{i}"
    #     for i in range(Config.long_title_desc_en_n_components)
    # ]
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

MlflowUtil.start_run(models)
MlflowUtil.log_config()
mlflow.log_param("model", models)
mlflow.log_param("features", "\n,".join(features))
mlflow.log_param("cat_features", "\n,".join(cat_features))
if "lgbm" in models:
    mlflow.lightgbm.autolog()
rmsles = []
for fold, (train_idx, valid_idx) in enumerate(folds):
    print(f"------------------------ fold {fold} -----------------------")
    _train_df = raw.train.loc[train_idx]
    _valid_df = raw.train.loc[valid_idx]

    if models in ["lgbm+cat", "lgbm"]:
        lgb_train_dataset = lgb.Dataset(_train_df[features], _train_df["likes_log"])
        lgb_valid_dataset = lgb.Dataset(_valid_df[features], _valid_df["likes_log"])
        lgb_model = lgb.train(
            Config.lgb_params,
            lgb_train_dataset,
            num_boost_round=1500,
            valid_sets=[lgb_train_dataset, lgb_valid_dataset],
            verbose_eval=50,
            early_stopping_rounds=300,
            categorical_feature=cat_features,
        )
        y_pred_lgb = np.expm1(lgb_model.predict(_valid_df[features]))

    if models in ["lgbm+cat", "cat"]:
        cat_train_dataset = Pool(
            _train_df[features], _train_df["likes_log"], cat_features=cat_features
        )
        cat_valid_dataset = Pool(
            _valid_df[features], _valid_df["likes_log"], cat_features=cat_features
        )
        cat_model = CatBoostRegressor(**Config.cat_params, iterations=3500)
        cat_model.fit(
            cat_train_dataset,
            verbose_eval=100,
            eval_set=[cat_valid_dataset],
            early_stopping_rounds=200,
        )
        y_pred_cat = np.expm1(cat_model.predict(_valid_df[features]))

    if models == "lgbm+cat":
        y_pred = (y_pred_cat + y_pred_lgb) / 2
    elif models == "lgbm":
        y_pred = y_pred_lgb
    elif models == "cat":
        y_pred = y_pred_cat

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
if "lgbm" in models:
    lgb.plot_importance(lgb_model, figsize=(16, 16))
    plt.show()
mlflow.end_run()

# %%
cat_train_dataset = Pool(
    raw.train[features], raw.train["likes_log"], cat_features=cat_features
)
lgb_train_dataset = lgb.Dataset(raw.train[features], raw.train["likes_log"])
cat_model = CatBoostRegressor(**Config.cat_params, iterations=2000)
cat_model.fit(
    cat_train_dataset,
    verbose_eval=100,
    eval_set=[cat_train_dataset],
)
lgb_model = lgb.train(
    Config.lgb_params,
    lgb_train_dataset,
    num_boost_round=700,
    verbose_eval=50,
    valid_sets=[lgb_train_dataset],
    categorical_feature=cat_features,
)
test_pred_cat = np.expm1(cat_model.predict(raw.test[features]))
test_pred_lgb = np.expm1(lgb_model.predict(raw.test[features]))
test_pred = (test_pred_cat + test_pred_lgb) / 2
test_pred[test_pred < 0] = 0
raw.sample_submission["likes"] = test_pred
raw.sample_submission.to_csv(Path.cwd() / "output" / "exp012_1.csv", index=False)

# %%


def generate_img(
    df: pd.DataFrame,
    object_id: str,
    img_width: int,
    img_height: int,
    save_dir: Optional[str],
    imshow: bool = False,
) -> np.ndarray:
    img = np.zeros((img_width, img_height, 3))
    df = df.sort_values("ratio", ascending=False)

    total_ratio = 0
    for row in df.itertuples():
        width_start = int(img_width * (total_ratio))
        width_end = int(img_width * (total_ratio + row.ratio))

        img[:, width_start:width_end, 0] = row.color_b
        img[:, width_start:width_end, 1] = row.color_g
        img[:, width_start:width_end, 2] = row.color_r

        total_ratio += row.ratio

    img = img.astype(np.int64)

    if save_dir is not None:
        save_path = os.path.join(save_dir, f"{object_id}.jpg")
        cv2.imwrite(save_path, img)

    if imshow:
        # plt.imshowの色順序がRGBなのでBGR->RGBに変換
        img_rgb = img[..., ::-1]
        plt.imshow(img_rgb)
        plt.show()

    return img


group = raw.palette.groupby("object_id")
for (object_id, df) in tqdm(group, total=len(group)):
    generate_img(
        df,
        object_id,
        Config.img_width,
        Config.img_height,
        str(Config.data_dir / "images"),
        imshow=False,
    )

# %%
raw.train.sort_values("likes", ascending=False)[
    ["likes", "title_lang", "title", "principal_or_first_maker", "dating_year_late"]
][140:160]

# %%
raw.palette.groupby("object_id")["ratio"].agg(["count", "sum"])

# %%
raw.palette.query("object_id == '0012765f7a97ccc3e9e9'").sort_values(
    "ratio", ascending=False
).reset_index(drop=True)

# %%
raw.train["title_len_cut"] = pd.qcut(raw.train["title_len"], 10)
raw.train.groupby("title_len_cut")["likes_log"].agg(["count", "mean"])

# %%
s = "the"
raw.train[f"title_contains_{s}"] = raw.train["title"].str.lower().str.contains(s)
raw.train.groupby([f"title_contains_{s}"])["likes_log"].agg(["count", "mean"])

# %%
raw.train["title_capital_word_num"] = raw.train["title"].map(
    lambda s: sum([1 if word[0].isupper() else 0 for word in s.split()])
)
raw.train.groupby(["title_capital_word_num"])["likes_log"].agg(["count", "mean"])

# %%
raw.train.groupby("principal_or_first_maker")["likes_log"].agg(
    ["count", "mean", "var"]
).sort_values("count", ascending=False)[:20]

# %%
raw.train.query("principal_or_first_maker == 'Aegidius Sadeler'").sort_values(
    "likes", ascending=True
)[
    [
        "likes",
        "likes_log",
        "title_lang",
        "title",
        "principal_or_first_maker",
        "dating_year_early",
        "dating_year_late",
        "dating_presenting_date",
    ]
].reset_index(
    drop=True
)[
    :20
]
