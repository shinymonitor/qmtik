data_file = open("mnist_784.csv", "r")
header = data_file.readline()

train_file = open("mnist_784_train", "wb")
infer_file = open("mnist_784_infer", "wb")

def write_sample(line, f):
    sample = list(map(int, line.strip().split(",")))
    pixels, label = sample[:-1], sample[-1]
    pixels = [max(-128, min(127, p - 128)) for p in pixels]
    out = [0] * 10
    out[label] = 127
    for p in pixels:
        f.write(int.to_bytes(p, 1, "big", signed=True))
    for o in out:
        f.write(int.to_bytes(o, 1, "big"))
count = 0
for line in data_file:
    if count < int(0.8 * 70000):
        write_sample(line, train_file)
    elif count < 70000:
        write_sample(line, infer_file)
    else:
        break
    count += 1
train_file.close()
infer_file.close()
data_file.close()
print("Train samples:", int(0.8 * 70000))
print("Infer samples:", int(0.2 * 70000))
