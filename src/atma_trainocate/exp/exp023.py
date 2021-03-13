# %%
import colorsys
import contextlib
import functools
import math
import os
import pickle
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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

    @staticmethod
    def get_metric() -> Tuple[float, float, float]:
        t = time.time()
        p = psutil.Process(os.getpid())
        m: float = p.memory_info()[0] / 2.0 ** 30
        per: float = psutil.virtual_memory().percent
        return t, m, per


class TimeUtil:
    @staticmethod
    @contextlib.contextmanager
    def timer(name: str):
        t0, m0, p0 = GlobalUtil.get_metric()
        print(f"[{name}] start [{m0:.1f}GB({p0:.1f}%)]")
        yield
        t1, m1, p1 = GlobalUtil.get_metric()
        delta = m1 - m0
        sign = "+" if delta >= 0 else "-"
        delta = math.fabs(delta)
        print(
            f"[{name}] done [{m1:.1f}GB({p1:.1f}%)({sign}{delta:.3f}GB)] {t1 - t0:.4f} s"
        )

    @staticmethod
    def timer_wrapper(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            t0, m0, p0 = GlobalUtil.get_metric()
            print(f"[{func.__name__}] start [{m0:.1f}GB({p0:.1f}%)]")
            value = func(*args, **kwargs)
            t1, m1, p1 = GlobalUtil.get_metric()
            delta = m1 - m0
            sign = "+" if delta >= 0 else "-"
            delta = math.fabs(delta)
            print(
                f"[{func.__name__}] done [{m1:.1f}GB({p1:.1f}%)({sign}{delta:.3f}GB)] {t1 - t0:.4f} s"
            )
            return value

        return wrapper


class FileUtil:
    @staticmethod
    def read_csv(filepath: Union[str, Path], verbose: bool = True):
        if verbose:
            with TimeUtil.timer(f"Read {str(filepath)}"):
                return pd.read_csv(filepath)
        return pd.read_csv(filepath)


class TestUtil:
    @staticmethod
    def assert_any(actual: Any, expect: Any):
        if actual != actual or expect != expect:
            assert (
                actual != actual and expect != expect
            ), f"Expect: {expect} Actual: {actual}"
        else:
            assert type(actual) == type(
                expect
            ), f"Expect type: {type(expect)}, Actual type: {type(actual)}"
            assert actual == expect, (
                "Expect: {} Actual: {}, difference: {}".format(
                    expect,
                    actual,
                    set(actual).union(set(expect))
                    - set(actual).intersection(set(expect)),
                )
                if isinstance(actual, list)
                else f"Expect: {expect} Actual: {actual}"
            )

    @staticmethod
    def test_raw(func: Callable):
        @functools.wraps(func)
        def wrapper(raw, *args, **kwargs):
            train_len_before = len(raw.train)
            test_len_before = len(raw.test)
            value = func(raw, *args, **kwargs)
            train_len_after = len(raw.train)
            test_len_after = len(raw.test)
            TestUtil.assert_any(train_len_after, train_len_before)
            TestUtil.assert_any(test_len_after, test_len_before)
            return value

        return wrapper


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


class Load:
    @staticmethod
    def load_raw() -> RawData:
        return RawData(
            train=FileUtil.read_csv(Config.data_dir / "train.csv"),
            test=FileUtil.read_csv(Config.data_dir / "test.csv"),
            color=FileUtil.read_csv(Config.data_dir / "color.csv"),
            historical_person=FileUtil.read_csv(
                Config.data_dir / "historical_person.csv"
            ),
            maker=FileUtil.read_csv(Config.data_dir / "maker.csv"),
            material=FileUtil.read_csv(Config.data_dir / "material.csv"),
            object_collection=FileUtil.read_csv(
                Config.data_dir / "object_collection.csv"
            ),
            palette=FileUtil.read_csv(Config.data_dir / "palette.csv"),
            principal_maker_occupation=FileUtil.read_csv(
                Config.data_dir / "principal_maker_occupation.csv"
            ),
            principal_maker=FileUtil.read_csv(Config.data_dir / "principal_maker.csv"),
            production_place=FileUtil.read_csv(
                Config.data_dir / "production_place.csv"
            ),
            technique=FileUtil.read_csv(Config.data_dir / "technique.csv"),
            colorspace_selected_pca=FileUtil.read_csv(
                Config.data_dir / "colorspace_selected_pca.csv"
            ),
            sample_submission=FileUtil.read_csv(
                Config.data_dir / "atmacup10__sample_submission.csv"
            ),
        )


class FE:
    @staticmethod
    @TimeUtil.timer_wrapper
    @TestUtil.test_raw
    def likes_log(raw: RawData) -> RawData:
        raw.train["likes_log"] = raw.train["likes"]
        return raw

    @staticmethod
    @TimeUtil.timer_wrapper
    @TestUtil.test_raw
    def sub_title(raw: RawData) -> RawData:
        def _sub_title(df: pd.DataFrame) -> pd.DataFrame:
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

        raw.train = _sub_title(raw.train)
        raw.test = _sub_title(raw.test)
        return raw

    @staticmethod
    @TimeUtil.timer_wrapper
    @TestUtil.test_raw
    def historical_person(raw: RawData) -> RawData:
        _counts = raw.historical_person["name"].value_counts()
        _df = raw.historical_person[
            raw.historical_person["name"].isin(_counts[_counts > 20].index.tolist())
        ]
        _df = pd.crosstab(_df["object_id"], _df["name"])
        for col in _df.columns:
            raw.train[f"historical_person_is_{col}"] = raw.train["object_id"].map(
                _df[col]
            )
            raw.test[f"historical_person_is_{col}"] = raw.test["object_id"].map(
                _df[col]
            )
        return raw

    @staticmethod
    @TimeUtil.timer_wrapper
    @TestUtil.test_raw
    def main(raw: RawData) -> RawData:
        raw = FE.likes_log(raw)
        raw = FE.sub_title(raw)
        raw = FE.historical_person(raw)
        return raw


GlobalUtil.seed_everything(Config.seed)
raw = Load.load_raw()

# %%
raw = FE.main(raw)

# %%
