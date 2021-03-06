# %%
import math
import os
import re
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil
import seaborn as sns
from catboost import CatBoostRegressor, Pool
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
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
    root_dir = Path.cwd().parents[2]
    data_dir = Path.cwd().parents[2] / "data/atmacup10_dataset"
    long_title_n_components = 3
    lgb_num_boost_round = 100
    lgb_params = {
        "num_leaves": 32,
        "min_data_in_leaf": 64,
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
        "num_threads": 6,
    }
    cat_params = {
        "min_data_in_leaf": 64,
        "learning_rate": 0.05,
        "reg_lambda": 0.3,
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

# %%


def label_encoding(
    label_encs: Dict[str, LabelEncoder], raw: RawData, col: str
) -> Tuple[Dict[str, LabelEncoder], RawData]:
    raw.train[col] = raw.train[col].fillna("nan")
    raw.test[col] = raw.test[col].fillna("nan")
    label_encs[col].fit(list(set(raw.train[col].tolist() + raw.test[col].tolist())))
    raw.train[col] = label_encs[col].transform(raw.train[col])
    raw.test[col] = label_encs[col].transform(raw.test[col])
    return label_encs, raw


def sub_title_to_number(s: str) -> float:
    number = float(re.findall(r"\d{1,10}", s)[0])
    if "mm" in s:
        return number
    elif "cm" in s:
        return number * 10
    else:
        return number


def _fe_sub_title(df: pd.DataFrame) -> pd.DataFrame:
    for key in ["h", "w", "d"]:
        df[key] = (
            df["sub_title"]
            .str.findall(key + r"\ \d{1,10}..")
            .map(
                lambda l: l[0].replace(f"{key} ", "")
                if isinstance(l, list) and len(l) > 0
                else 0
            )
            .astype("str")
        )
        df[key] = df[key].map(sub_title_to_number)
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
    with TimeUtil.timer("color_r"):
        color_r = (
            raw.palette.groupby("object_id")
            .apply(lambda row: (row.ratio * row.color_r).sum())
            .to_dict()
        )
    with TimeUtil.timer("color_g"):
        color_g = (
            raw.palette.groupby("object_id")
            .apply(lambda row: (row.ratio * row.color_g).sum())
            .to_dict()
        )
    with TimeUtil.timer("color_b"):
        color_b = (
            raw.palette.groupby("object_id")
            .apply(lambda row: (row.ratio * row.color_b).sum())
            .to_dict()
        )
    raw.train["color_r"] = raw.train["object_id"].map(color_r)
    raw.train["color_g"] = raw.train["object_id"].map(color_g)
    raw.train["color_b"] = raw.train["object_id"].map(color_b)
    raw.test["color_r"] = raw.test["object_id"].map(color_r)
    raw.test["color_g"] = raw.test["object_id"].map(color_g)
    raw.test["color_b"] = raw.test["object_id"].map(color_b)
    return raw


def fe_long_title_pca(raw: RawData, n_components: int) -> RawData:
    train_long_title_encoded = np.load(
        Config.root_dir / "data/train_long_title_encoded.npy"
    )
    test_long_title_encoded = np.load(
        Config.root_dir / "data/test_long_title_encoded.npy"
    )
    concated = np.concatenate((train_long_title_encoded, test_long_title_encoded))
    pca = PCA(n_components=n_components).fit(concated)
    train_long_title_pca = pca.transform(train_long_title_encoded)
    test_long_title_pca = pca.transform(test_long_title_encoded)
    raw.train.loc[
        :, [f"long_title_pca_{i}" for i in range(n_components)]
    ] = train_long_title_pca
    raw.test.loc[
        :, [f"long_title_pca_{i}" for i in range(n_components)]
    ] = test_long_title_pca
    return raw


def fe(raw: RawData, long_title_n_components: int) -> RawData:
    raw = fe_sub_title(raw)
    raw = fe_historical_person(raw)
    raw = fe_material(raw)
    raw = fe_color(raw)
    raw = fe_object_collection(raw)
    raw = fe_technique(raw)
    raw = fe_production_place(raw)
    raw = fe_palette(raw)
    raw = fe_long_title_pca(raw, long_title_n_components)
    return raw


raw.train["likes_log"] = np.log1p(raw.train["likes"])
raw = fe(raw, Config.long_title_n_components)


# %%
features = [
    "h",
    "w",
    "d",
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
] + [f"long_title_pca_{i}" for i in range(Config.long_title_n_components)]

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
    ]
]


label_encs = {col: LabelEncoder() for col in cat_features}
for col in cat_features:
    label_encs, raw = label_encoding(label_encs, raw, col)

folds = StratifiedKFold(
    n_splits=Config.n_splits, shuffle=True, random_state=Config.seed
).split(raw.train["object_id"], raw.train["likes"])
rmsles = []
for fold, (train_idx, valid_idx) in enumerate(folds):
    print(f"------------------------ fold {fold} -----------------------")
    _train_df = raw.train.loc[train_idx]
    _valid_df = raw.train.loc[valid_idx]
    # train_dataset = lgb.Dataset(_train_df[features], _train_df["likes_log"])
    # valid_dataset = lgb.Dataset(_valid_df[features], _valid_df["likes_log"])
    # model = lgb.train(
    #     Config.lgb_params,
    #     train_dataset,
    #     num_boost_round=1000,
    #     valid_sets=[train_dataset, valid_dataset],
    #     verbose_eval=50,
    #     early_stopping_rounds=200,
    #     categorical_feature=cat_features,
    # )
    train_dataset = Pool(
        _train_df[features], _train_df["likes_log"], cat_features=cat_features
    )
    valid_dataset = Pool(
        _valid_df[features], _valid_df["likes_log"], cat_features=cat_features
    )
    model = CatBoostRegressor(**Config.cat_params)
    model.fit(
        train_dataset,
        verbose_eval=100,
        eval_set=[train_dataset, valid_dataset],
        early_stopping_rounds=200,
    )
    y_pred = np.expm1(model.predict(_valid_df[features]))
    y_pred[y_pred < 0] = 0
    y_true = _valid_df["likes"].values
    rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))
    rmsles.append(rmsle)
    print(f"------------------------ fold {fold} -----------------------")
    print(f"------------------- rmsle {rmsle} -----------------------")
    print()

print("")
print(f"------------------- average rmsle {np.mean(rmsles)} -----------------------")

# %%
train_dataset = Pool(
    raw.train[features], raw.train["likes_log"], cat_features=cat_features
)
model = CatBoostRegressor(**Config.cat_params)
model.fit(
    train_dataset, verbose_eval=100, eval_set=[train_dataset],
)
test_pred = np.expm1(model.predict(raw.test[features]))
test_pred[test_pred < 0] = 0
raw.sample_submission["likes"] = test_pred
raw.sample_submission.to_csv(Path.cwd() / "output" / "exp003_1.csv", index=False)

