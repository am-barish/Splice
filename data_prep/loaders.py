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
from data_prep.wildcam import WildCamDataset
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


def amazon_sample_config(config_name, n_samples=10000):
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
        n_val = int(0.3 * len(idx))
        df.loc[idx[:n_val], "split"] = "val"

    df.to_parquet('reviews_raw.parquet')
    return df
