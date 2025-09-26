# https://www.kaggle.com/datasets/aadeshkoirala/mnist-784

CSV="mnist_784.csv"
I=784
O=10
N=70000

TRAIN_SPLIT=0.8
csv_file = open(CSV, "r")
csv_file.readline()
train_file = open(CSV[:-4]+"_train", "wb")
test_file = open(CSV[:-4]+"_test", "wb")
def write_sample(line, file):
    sample = list(map(int, line.strip().split(",")))
    pixels, label = sample[:-1], sample[-1]
    pixels = [max(-128, min(127, p - 128)) for p in pixels]
    out = [0] * O
    out[label] = 127
    for p in pixels:
        file.write(int.to_bytes(p, 1, "big", signed=True))
    for o in out:
        file.write(int.to_bytes(o, 1, "big"))
count=0
train_count = 0
test_count = 0
for line in csv_file:
    if count < int(TRAIN_SPLIT*N):
        write_sample(line, train_file)
        train_count += 1
    elif count < N:
        write_sample(line, test_file)
        test_count += 1
    else:
        break
    count+=1
train_file.close()
test_file.close()
csv_file.close()
if count!=N or train_count!=round(TRAIN_SPLIT*N) or test_count!=round((1-TRAIN_SPLIT)*N):
    print("FAILED")
    exit()
print("Train samples:", train_count)
print("Test samples:", test_count)