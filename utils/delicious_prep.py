import pandas as pd

# Data index start from 1
with open('../data/Delicious/Delicious_data.txt') as f:
    content = f.readlines()

# # you may also want to remove whitespace characters like `\n` at the end of each line
# content = [x.strip() for x in content] 

data = pd.read_csv('../data/Delicious/delicious_trSplit.txt', sep=" ", header=None)
data = data.to_numpy()

training_data=open('../data/Delicious/train.txt','w')

for i in range(len(data[:,0])):
	training_data.write(content[data[i][0]])
training_data.close()

data = pd.read_csv('../data/Delicious/delicious_tstSplit.txt', sep=" ", header=None)
data = data.to_numpy()

test_data=open('../data/Delicious/test.txt','w')

for i in range(len(data[:,0])):
	test_data.write(content[data[i][0]])
test_data.close()