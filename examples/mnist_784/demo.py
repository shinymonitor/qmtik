#!/usr/bin/env python3
import sys

def read_ppm(filepath):
    with open(filepath, 'r') as f:
        magic = f.readline().strip()
        if magic != 'P3':
            print(f"Error: Not a P3 PPM file (found {magic})")
            sys.exit(1)
        line = f.readline().strip()
        while line.startswith('#'):
            line = f.readline().strip()
        width, height = map(int, line.split())
        if width * height != 784:
            print(f"Error: Image must be 784 pixels (28x28), got {width}x{height} = {width*height}")
            sys.exit(1)
        max_val = int(f.readline().strip())
        pixels = []
        for line in f:
            pixels.extend(map(int, line.split()))
        grayscale = [pixels[i] for i in range(0, len(pixels), 3)]
        if len(grayscale) != 784:
            print(f"Error: Expected 784 pixels, got {len(grayscale)}")
            sys.exit(1)
        return grayscale

def write_sample_pair(pixels, label, output_file):
    I = 784
    O = 10
    pixels_int8 = [max(-128, min(127, p - 128)) for p in pixels]
    output = [0] * O
    output[label] = 127
    with open(output_file, 'wb') as f:
        for p in pixels_int8:
            f.write(int.to_bytes(p, 1, "big", signed=True))
        for o in output:
            f.write(int.to_bytes(o, 1, "big", signed=False))

def main():
    if len(sys.argv) != 3:
        print("Usage: <path_to_ppm> <label_number>")
        sys.exit(1)
    ppm_path = sys.argv[1]
    label = int(sys.argv[2])
    if label < 0 or label > 9:
        print("Error: Label must be between 0 and 9")
        sys.exit(1)
    print(f"Reading {ppm_path}...")
    pixels = read_ppm(ppm_path)
    output_file = ppm_path.rsplit('.', 1)[0] + "_sample"
    print(f"Writing sample pair to {output_file}...")
    write_sample_pair(pixels, label, output_file)
    print(f"Done! Created sample for digit {label}")
    print(f"Input: 784 bytes (int8)")
    print(f"Output: 10 bytes (one-hot encoded)")

if __name__ == "__main__":
    main()