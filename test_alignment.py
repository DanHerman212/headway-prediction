import os
# 1. Force CPU to avoid GPU initialization locks
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from data import DataExtractor, DataPreprocessor
from training import Trainer
from config import ModelConfig

# 2. Add the Main Guard (Required for TF Multiprocessing)
if __name__ == "__main__":
    os.environ['GCP_PROJECT_ID'] = 'realtime-headway-prediction'

    config = ModelConfig.from_env()
    config.lookback_steps = 20
    config.batch_size = 64

    print("Loading data...")
    extractor = DataExtractor(config)
    df_raw = extractor.extract()
    preprocessor = DataPreprocessor(config)
    df_preprocessed = preprocessor.preprocess(df_raw)
    preprocessor.save(df_preprocessed, 'data/X.csv')

    print("\nCreating datasets...")
    trainer = Trainer(config)
    trainer.load_data('data/X.csv')
    train_dataset, val_dataset, test_dataset = trainer.create_datasets()

    print("\n" + "="*70)
    print("INSPECT TRAIN DATASET")
    print("="*70)
    # This will now run without hanging
    trainer.inspect_batch(train_dataset, num_examples=3)
