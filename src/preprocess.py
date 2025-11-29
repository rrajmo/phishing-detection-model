import pandas as pd
import logging
from pathlib import Path
from typing import Tuple, Any
from paths import get_path
from similarity_url_index import get_maximum_similarity_url_index
from extract_features import (
    extract_url_length,
    extract_domain_length,
    is_domain_ip_address,
    extract_character_continuation_rate,
    extract_tld_length,
    extract_number_of_subdomains,
    extract_has_obfuscation,
    extract_number_of_obfuscated_characters,
    extract_obfuscation_ratio,
    extract_number_of_letters,
    extract_letter_ratio,
    extract_number_of_digits,
    extract_digit_ratio,
    extract_number_of_equals_sign,
    extract_number_of_question_mark,
    extract_number_of_ampersand,
    is_https
)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]

def get_config(config: dict) -> Tuple[Any, Any, Any]:
    paths = get_path(config)

    RAW_DATA_MENDELEY = paths["raw_data_Mendeley"]
    RAW_DATA_PHIUSIIL = paths["raw_data_PhiUSIIL"]
    PROCESSED_DATA_PATH = paths["processed_data"]

    if not RAW_DATA_MENDELEY.exists():
        raise FileNotFoundError(f"Data file not present at {RAW_DATA_MENDELEY}")
    if not RAW_DATA_PHIUSIIL.exists():
        raise FileNotFoundError(f"Data file not present at {RAW_DATA_PHIUSIIL}")
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    return RAW_DATA_MENDELEY, RAW_DATA_PHIUSIIL, PROCESSED_DATA_PATH

def load_raw_csv(file: str) -> pd.DataFrame:
    df = pd.read_csv(file)
    logger.info(f"Loaded data from {file}")
    return df

def rename_url_column(df: pd.DataFrame) -> pd.DataFrame:
    old_column = "url"
    new_column = "URL"
    df = df.rename(columns={old_column: new_column})
    logger.info(f"Renamed column '{old_column}' to column '{new_column}'")
    return df

def filter_urls(df: pd.DataFrame, minimum_length: int = 51) -> pd.DataFrame:
    df["URL_Length"] = df["URL"].astype(str).apply(extract_url_length)
    df_filtered = df[df["URL_Length"] >= minimum_length]
    df_filtered = df_filtered.drop(columns=["URL_Length"])
    logger.info(f"Filtered URLs with length greater than {minimum_length}")
    logger.info(f"The number of rows to process is {len(df_filtered)}")
    return df_filtered

def add_url_features(df: pd.DataFrame) -> pd.DataFrame:
    df["URLLength"] = df["URL"].astype(str).apply(extract_url_length)
    df["DomainLength"] = df["URL"].astype(str).apply(extract_domain_length)
    df["IsDomainIP"] = df["URL"].astype(str).apply(is_domain_ip_address)
    df["URLSimilarityIndex"] = df["URL"].astype(str).apply(get_maximum_similarity_url_index)
    df["CharContinuationRate"] = df["URL"].astype(str).apply(extract_character_continuation_rate)
    df["TLDLength"] = df["URL"].astype(str).apply(extract_tld_length)
    df["NoOfSubDomain"] = df["URL"].astype(str).apply(extract_number_of_subdomains)
    df["HasObfuscation"] = df["URL"].astype(str).apply(extract_has_obfuscation)
    df["NoOfObfuscatedChar"] = df["URL"].astype(str).apply(extract_number_of_obfuscated_characters)
    df["ObfuscationRatio"] = df["URL"].astype(str).apply(extract_obfuscation_ratio)
    df["NoOfLettersInURL"] = df["URL"].astype(str).apply(extract_number_of_letters)
    df["LetterRatioInURL"] = df["URL"].astype(str).apply(extract_letter_ratio)
    df["NoOfDegitsInURL"] = df["URL"].astype(str).apply(extract_number_of_digits)
    df["DegitRatioInURL"] = df["URL"].astype(str).apply(extract_digit_ratio)
    df["NoOfEqualsInURL"] = df["URL"].astype(str).apply(extract_number_of_equals_sign)
    df["NoOfQMarkInURL"] = df["URL"].astype(str).apply(extract_number_of_question_mark)
    df["NoOfAmpersandInURL"] = df["URL"].astype(str).apply(extract_number_of_ampersand)
    df["IsHTTPS"] = df["URL"].astype(str).apply(is_https)
    df["label"] = df["type"].str.lower().map({"legitimate": 0, "phishing": 1})
    df = df.drop(columns=["type"])
    logger.info("Extracted URL features")
    return df

def clean_current_data(df: pd.DataFrame, valid_columns: list) -> pd.DataFrame:
    extra_columns = [c for c in df.columns if c not in valid_columns]
    df = df.drop(columns=extra_columns)
    df = df.reindex(columns=valid_columns)
    logger.info("Removed extra columns from current data")
    return df

def combine_datasets(df_phiusiil: pd.DataFrame, df_mendeley: pd.DataFrame) -> pd.DataFrame:
    combined_df = pd.concat([df_phiusiil, df_mendeley], ignore_index=True)
    logger.info("Combined both datasets")
    return combined_df

def save_processed_csv(file: str, df: pd.DataFrame) -> None:
    logger.info(f"Saved final dataset to {file}")
    df.to_csv(file, index=False)

def process_dataset(config: dict) -> None:
    RAW_DATA_MENDELEY, RAW_DATA_PHIUSIIL, PROCESSED_DATA_PATH = get_config(config)

    if PROCESSED_DATA_PATH.exists():
        logger.info(f"{PROCESSED_DATA_PATH} already exists")
        return

    df_mendeley = load_raw_csv(RAW_DATA_MENDELEY)
    df_mendeley = rename_url_column(df_mendeley)

    df_mendeley_filtered = filter_urls(df_mendeley, minimum_length=51)
    df_mendeley_features = add_url_features(df_mendeley_filtered)

    df_phiusiil = load_raw_csv(RAW_DATA_PHIUSIIL)

    valid_columns = [
        "URL",
        "URLLength",
        "DomainLength",
        "IsDomainIP",
        "URLSimilarityIndex",
        "CharContinuationRate",
        "TLDLength",
        "NoOfSubDomain",
        "HasObfuscation",
        "NoOfObfuscatedChar",
        "ObfuscationRatio",
        "NoOfLettersInURL",
        "LetterRatioInURL",
        "NoOfDegitsInURL",
        "DegitRatioInURL",
        "NoOfEqualsInURL",
        "NoOfQMarkInURL",
        "NoOfAmpersandInURL",
        "IsHTTPS",
        "label",
    ]

    df_phiusiil_cleaned = clean_current_data(df_phiusiil, valid_columns)
    combined_df = combine_datasets(df_phiusiil_cleaned, df_mendeley_features)

    save_processed_csv(PROCESSED_DATA_PATH, combined_df)
