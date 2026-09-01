"""
Train all ML models sequentially.
Usage: python scripts/train_all_models.py [--data path/to/dataset.csv]
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from loguru import logger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=None, help='Path to CSV dataset')
    args = parser.parse_args()
    data_path = args.data

    logger.info("=" * 60)
    logger.info("TRAINING ALL ANOMALY DETECTION MODELS")
    logger.info("=" * 60)

    results = {}

    logger.info("\n[1/4] Training Random Forest...")
    from ml.train_random_forest import train_random_forest
    results['random_forest'] = train_random_forest(data_path=data_path)

    logger.info("\n[2/4] Training XGBoost...")
    from ml.train_xgboost import train_xgboost
    results['xgboost'] = train_xgboost(data_path=data_path)

    logger.info("\n[3/4] Training Isolation Forest...")
    from ml.train_isolation_forest import train_isolation_forest
    results['isolation_forest'] = train_isolation_forest(data_path=data_path)

    logger.info("\n[4/4] Training Autoencoder...")
    from ml.train_autoencoder import train_autoencoder
    results['autoencoder'] = train_autoencoder(data_path=data_path)

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE — Summary:")
    logger.info("=" * 60)
    for model, metrics in results.items():
        if metrics:
            f1 = metrics.get('f1_score', 0)
            acc = metrics.get('accuracy', 0)
            logger.info(f"  {model:25s} → Accuracy: {acc:.3f} | F1: {f1:.3f}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
