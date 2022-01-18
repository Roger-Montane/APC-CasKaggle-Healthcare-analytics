import pandas as pd

# Function to read csv files and save them as a pandas structure
def load_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter=',')
    return dataset

# Load train data
train = load_dataset('../data/train_data.csv')
train_data = train.copy(deep=True)
print(train_data.shape)

# Load test data
test = load_dataset('../data/test_data.csv')
test_data = test.copy(deep=True)
print(test_data.shape)