# Steps to Set Up and Run the Project:

### 1. Create a virtual environment

`python -m venv venv`

### 2. Activate virtual environment

Windows (Powershell):
`venv\Scripts\Activate.ps1`

Windows (Command Prompt):
`venv\Scripts\Activate.bat`

Mac/Linux:
`source venv/bin/activate`

### 3. Install dependencies

`pip install -r requirements.txt`

### 4. Run project

```
cd src
python main.py
```

## Dataset Sources

Download the dataset from the publication:

Prasad, A., & Chandra, S. (2023). PhiUSIIL: A diverse security profile empowered phishing URL detection framework based on similarity index and incremental learning. Computers & Security, 103545. doi: https://doi.org/10.1016/j.cose.2023.103545

This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license. This license allows sharing and adapting the dataset for any purpose, as long as appropriate credit is given to the original authors.

Save the file as `data\raw\PhiUSIIL_Phishing_URL_Dataset.csv`

Download the dataset from the dataset:

KAITHOLIKKAL, JISHNU K S; B, Arthi  (2024), “Phishing URL dataset”, Mendeley Data, V1, doi: 10.17632/vfszbj9b36.1

This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license. This license allows sharing and adapting the dataset for any purpose, as long as appropriate credit is given to the original authors.

Save the file as `data\raw\Mendeley_Dataset.csv`

Download the "Top 10 Million Websites" dataset from the Open PageRank Initiative

Filter the dataset to include only the top 1000 URLs and remove duplicates based on domain names

Save the file as `domains\top1thousanddomains.csv`

## Configuration Options

Users can customize settings in `config\config.yaml` file to control model training and evaluation.

### 1. Train Test Split

```
train_test_split:
  test_size: 0.2
  random_state: 42
```

### 2. Cross Validation

```
cross_validation:
  splits: 5
```

### 3. Model Parameters

```
model_parameters:
  objective: binary
  metric: binary_logloss
  random_state: 42
  verbosity: -1
```
