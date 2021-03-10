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
    material_n_components = 10
    technique_n_components = 5
    color_n_components = 3
    cat_aggs = ["count"]
    object_collections = ["prints", "paintings", "other"]
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


# done technique -> 数と単語ベース
# done material -> 数と単語ベース
# historical_person
# maker
# principal_maker
# principal_maker_occupation


# %%
raw.train.groupby("object_collection")["likes_log"].agg(["count", "mean"]).sort_values(
    "mean", ascending=False
)
