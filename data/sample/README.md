# Sample Data

Place your sample CSV or PCAP files here for testing.

## Supported formats
- CSV: network flow files (CICIDS2017, UNSW-NB15, NSL-KDD, etc.)
- PCAP / PCAPNG: packet capture files

## Generate synthetic data
```bash
python -c "
from ml.data_loader import generate_sample_dataset
df, labels = generate_sample_dataset(1000)
df['label'] = labels
df.to_csv('data/sample/synthetic_flows.csv', index=False)
print('Generated data/sample/synthetic_flows.csv')
"
```
