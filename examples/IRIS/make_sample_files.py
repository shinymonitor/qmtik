# https://www.kaggle.com/datasets/mzeeshan786/iris-dataset

import random

CSV="IRIS.csv"
I=4
O=3
N=150

TRAIN_SPLIT=0.8
csv_file = open(CSV, "r")
csv_file.readline()
train_file = open(CSV[:-4]+"_train", "wb")
test_file = open(CSV[:-4]+"_test", "wb")
samples=[]
def store_sample(line):
    sample = line.strip().split(",")
    prop, label = sample[:-1], sample[-1]
    prop = [max(-128, min(127, round((float(p) / 8.0) * 255.0 - 128.0))) for p in prop]
    out = [0] * O
    labels=["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    out[labels.index(label)] = 127
    samples.append([prop, out])

count=0
train_count = 0
test_count = 0
for line in csv_file:
    store_sample(line)

while samples:
    i=random.randint(0, len(samples)-1)
    if count < int(TRAIN_SPLIT*N):
        for p in samples[i][0]:
            train_file.write(int.to_bytes(p, 1, "big", signed=True))
        for o in samples[i][1]:
            train_file.write(int.to_bytes(o, 1, "big"))
        train_count += 1
    elif count < N:
        for p in samples[i][0]:
            test_file.write(int.to_bytes(p, 1, "big", signed=True))
        for o in samples[i][1]:
            test_file.write(int.to_bytes(o, 1, "big"))
        test_count += 1
    else:
        break
    samples.pop(i)
    count+=1
train_file.close()
test_file.close()
csv_file.close()
if count!=N or train_count!=round(TRAIN_SPLIT*N) or test_count!=round((1-TRAIN_SPLIT)*N):
    print("FAILED")
    exit()
print("Train samples:", train_count)
print("Test samples:", test_count)
