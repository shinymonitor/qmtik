# https://www.kaggle.com/datasets/zalando-research/fashionmnist

TRAIN_CSV="fashion-mnist_train.csv"
TEST_CSV="fashion-mnist_test.csv"
I=784
O=10

TRAIN_N=60000
TEST_N=10000

def write_sample(line, file):
    sample = list(map(int, line.strip().split(",")))
    pixels, label = sample[1:], sample[0]
    pixels = [max(-128, min(127, p - 128)) for p in pixels]
    out = [0] * O
    out[label] = 127
    for p in pixels:
        file.write(int.to_bytes(p, 1, "big", signed=True))
    for o in out:
        file.write(int.to_bytes(o, 1, "big"))

train_csv_file = open(TRAIN_CSV, "r")
train_csv_file.readline()
train_file = open(TRAIN_CSV[:-4], "wb")
train_count = 0
for line in train_csv_file:
    write_sample(line, train_file)
    train_count += 1
train_file.close()

test_csv_file = open(TEST_CSV, "r")
test_csv_file.readline()
test_file = open(TEST_CSV[:-4], "wb")
test_count = 0
for line in test_csv_file:
    write_sample(line, test_file)
    test_count += 1
test_file.close()

if train_count!=TRAIN_N or test_count!=TEST_N:
    print("FAILED")
    exit()
print("Train samples:", train_count)
print("Test samples:", test_count)
