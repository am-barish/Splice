from folktables import ACSDataSource, ACSPublicCoverage, BasicProblem, adult_filter, ACSIncome
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import TargetEncoder, OneHotEncoder, LabelEncoder, OrdinalEncoder
from category_encoders import BinaryEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
from pathlib import Path
import pandas as pd

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from wildcam_data import WildCamDataset
from datasets import load_dataset
from datasets import concatenate_datasets

year_list = ['2014', '2015', '2016', '2017', '2018']

state_list = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI',
              'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI',
              'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC',
              'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
              'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR']
ACSIncome56 = BasicProblem(
    features=[
        'AGEP',
        'COW',
        'SCHL',
        'MAR',
        'OCCP',
        'POBP',
        'RELP',
        'WKHP',
        'SEX',
        'RAC1P',
    ],
    target='PINCP',
    target_transform=lambda x: x > 56000,
    group='RAC1P',
    preprocess=adult_filter,
    postprocess=lambda x: np.nan_to_num(x, -1),
)

acs_dict = {"ACSIncome" : ACSIncome, "ACSPublicCoverage" : ACSPublicCoverage, "ACSIncome56" : ACSIncome56}


def race_encode(x):
    if x==1:
        return 1.0
    return 2.0

def load_classification_dataset(dataset_names):
    data = dict()
    for name in dataset_names:
        if (name == "ACSIncome" or name == "ACSPublicCoverage" or name == "ACSIncome56"):
            df = pd.DataFrame()
            for year in tqdm(year_list):
                data_source = ACSDataSource(survey_year=year, horizon='1-Year', survey='person')
                for state in tqdm(state_list):
                    state_data = data_source.get_data(states=[state], download = False)
                    temp_df = acs_dict[name].df_to_pandas(state_data)
                    l2 = [state]*len(temp_df[0])
                    l1 = [year]*len(temp_df[0])
                    x = pd.DataFrame([l1,l2],index=['Year','State']).T
                    test_df = pd.concat([x,temp_df[0],temp_df[1]],axis=1)
                    df = pd.concat([df,test_df],axis=0)
        data[name] = df
        if name == "ACSPublicCoverage":
            data[name]["RAC1P"] = data[name]["RAC1P"].apply(race_encode)
    return data

def load_regression_dataset(dataset_names):
    data = dict()
    for name in dataset_names:
        if name == "ACSTravelTime":
            df_orig = pd.DataFrame()
            for year in tqdm(year_list):
                data_source = ACSDataSource(survey_year=year, horizon='1-Year', survey='person')
                for state in tqdm(state_list):
                    state_data = data_source.get_data(states=[state], download=False)

                    df = state_data.copy()
                    df = df[df['AGEP'] > 16]
                    df = df[df['PWGTP'] >= 1]
                    df = df[df['ESR'] == 1]

                    variables = df[[
                            'AGEP',
                            'SCHL',
                            'MAR',
                            'SEX',
                            'DIS',
                            'ESP',
                            'MIG',
                            'RELP',
                            'RAC1P',
                            'PUMA',
                            'ST',
                            'CIT',
                            'OCCP',
                            'JWTR',
                            'POWPUMA',
                            'POVPIP',
                        ]]
                    postprocess = lambda x: np.nan_to_num(x, -1)
                    target_transform=lambda x: x
                    target="JWMNP"
                    group='RAC1P'
                    group_transform=lambda x: x

                    variables = pd.DataFrame(postprocess(variables.to_numpy()),
                                                    columns=variables.columns)
                    target = target_transform(df[target])
                    target = pd.DataFrame(target).reset_index(drop=True)

                    group = group_transform(df[group])
                    group = pd.DataFrame(group).reset_index(drop=True)

                    temp_df = variables, target, group

                    l2 = [state]*len(temp_df[0])
                    l1 = [year]*len(temp_df[0])
                    x = pd.DataFrame([l1,l2],index=['Year','State']).T
                    test_df = pd.concat([x,temp_df[0],temp_df[1]],axis=1)
                    df_orig = pd.concat([df_orig,test_df],axis=0)
        data[name] = df_orig.dropna()
    return data

def create_train_test_sources(data, dataset, split_type = "full"):
    train_sources = []
    test_source = pd.DataFrame()
    if split_type=="full":
        train_data, test_source = train_test_split(data, test_size=0.2, random_state=42)
    else:
        train_data = data
    sources = []
    if dataset == "Flights":
        for i in tqdm(range(len(train_data))):
            sources.append(train_data[i].drop(columns = ["OP_UNIQUE_CARRIER", "DEP_DELAY"]).reset_index(drop=True))
    else:
        for s in tqdm(state_list):
            sources.append(train_data[train_data["State"] == s])
    for s in sources:
        s = s.reset_index(drop=True)
        if split_type == "ood":
            train_sources.append(s.reset_index(drop = True))
        else:
            train_data, test_data = train_test_split(s, test_size=5000, random_state=1)
            train_sources.append(train_data.reset_index(drop=True))
            test_source = pd.concat([test_source, test_data])
    if split_type == "ood":
        test_source = train_sources[-1]
        train_sources = train_sources[:-1]
    return train_sources, test_source


def amazon_sample_config(config_name, n_samples=10000):
    print(f"Loading {config_name}...")
    ds = load_dataset("McAuley-Lab/Amazon-Reviews-2023", config_name, streaming=True)

    split_name = list(ds.keys())[0]
    category_stream = ds[split_name]

    shuffled = category_stream.shuffle(buffer_size=10000, seed=42)
    sampled = shuffled.take(n_samples)

    return list(sampled)


def amazon_review_setup():
    source_configs =[
    'raw_review_Books',
    'raw_review_Kindle_Store',
    'raw_review_Subscription_Boxes',
    'raw_review_Office_Products',
    'raw_review_Software',
    'raw_review_Electronics',
    'raw_review_Home_and_Kitchen',
    'raw_review_Clothing_Shoes_and_Jewelry',
    'raw_review_Sports_and_Outdoors',
    'raw_review_Tools_and_Home_Improvement',
    'raw_review_Beauty_and_Personal_Care',
    'raw_review_Health_and_Household',
    'raw_review_Toys_and_Games',
    'raw_review_Pet_Supplies',
    'raw_review_Patio_Lawn_and_Garden',
    'raw_review_Industrial_and_Scientific',
    'raw_review_Automotive',
    'raw_review_Musical_Instruments',
    'raw_review_Video_Games',
    'raw_review_Arts_Crafts_and_Sewing',
    'raw_review_Appliances',
    'raw_review_Grocery_and_Gourmet_Food',
    'raw_review_Handmade_Products',
    'raw_review_Baby_Products',
    'raw_review_CDs_and_Vinyl',
]


    target_configs = [
    'raw_review_Movies_and_TV',
    'raw_review_Magazine_Subscriptions',
    'raw_review_Handmade_Products'
]
    all_data = {}
    for config in source_configs + target_configs:
        all_data[config] = amazon_sample_config(config)

    df_list = []
    for config, samples in all_data.items():
        category = config[11:]
        for sample in samples:
            df_list.append({
                'category': category,
                'product_id': sample.get('product_id', sample.get('asin', 'unknown')),
                'review_body': sample.get('text', ''),
                'star_rating': sample.get('rating', 3),
                'label': 1 if sample.get('rating', 3) >= 4 else 0
            })

    df = pd.DataFrame(df_list)

    df["split"] = "train"
    for cat in ['Movies_and_TV','Magazine_Subscriptions','Handmade_Products']:
        cat_df = df[df["category"] == cat]
        idx = cat_df.index.to_numpy()
        rng = np.random.default_rng(42)
        rng.shuffle(idx)
        n_val = int(0.1 * len(idx))
        df.loc[idx[:n_val], "split"] = "val"

    print(df.groupby(["category", "split"]).size())
    df.to_parquet('reviews_raw.parquet')
    return df
