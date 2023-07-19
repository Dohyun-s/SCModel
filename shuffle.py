import random
data_file = '/home/minsu/CLIPP/training_data/pdb_2021aug02_sample/list.csv'
with open(data_file, 'r') as file:
    indices = file.read().splitlines()

# Split the indices into train and validation sets
random.shuffle(indices)
split_ratio = 0.05
split_index = int(len(indices) * split_ratio)

train_indices = indices[:split_index]

train_file = 'train_s'
with open(train_file, 'w') as file:
    for index in train_indices:
        file.write(index + '\n')

