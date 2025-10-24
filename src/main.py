import logging
from logger import initialize_logger
from config import load_config
from preprocess import process_dataset
from train import train_model
from evaluate import evaluate_model

if __name__ == "__main__":
    logger = initialize_logger(level=logging.INFO)
    config = load_config()
    process_dataset(config)
    train_model(config)
    evaluate_model(config)