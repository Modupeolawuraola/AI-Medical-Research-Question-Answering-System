import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import os
import pandas as pd
import zipfile
from huggingface_hub import HfApi

def setup_kaggle():
    try:
        api = KaggleApi()
        api.authenticate()
        print('Kaggle API authenticated')
        return api
    except Exception as e:
        print(f'Authentication Error: {e}')
        return None

def process_metadata():
    if os.path.exists('./data/covid19_subset.csv'):
        try:
            df = pd.read_csv('./data/covid19_subset.csv')
            return df
        except Exception as e:
            print(f"Error loading existing data: {e}")
            return None

    if os.path.exists('./data/metadata.zip'):
        with zipfile.ZipFile('/data/metadata.zip', 'r') as zip_re:
            zip_re.extractall('./data')
    else:
        api=setup_kaggle()
        if api:
            try:
                api.dataset_download_file(
                    'allen-institute-for-ai/CORD-19-research-challenge',
                    'metadata.csv',
                    path='./data')
                with zipfile.ZipFile('./data/metadata.csv.zip', 'r') as zip_ref:
                    zip_ref.extractall('./data')
            except Exception as e:
                print(f"Download Error: {e}")
                return None

    try:
        chunks = pd.read_csv('./data/metadata.csv.zip', chunksize= 1000)
        selected_data=pd.concat([next(chunks) for _ in range(5)])
        selected_data.to_csv('./data/covid19_subset.csv', index=False)
        return selected_data
    except Exception as e:
        print(f"Processing Error: {e}")
        return None

def test_hf_token():
    try:
        api=HfApi()
        identity = api.whoami()
        print(f"Successfully authenticated as :{identity}")
        return True
    except Exception as e:
        print(f"Token verification failed: {e}")
        return

