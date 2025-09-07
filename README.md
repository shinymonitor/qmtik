# Quantized Model Training and Inference Kit

A minimal, dependency-free implementation of a quantized neural network designed for embedded systems and resource constrained environments. This implementation uses 8-bit integer quantization for both weights and activations, enabling efficient inference on microcontrollers and edge devices.

## Features
- No dependencies, runs on any C99 compatible compiler
- INT8 weights and activations for maximum memory efficiency
- Easy to modify network topology via config.h
- Multiple activation, output processing and cost functions
- Optimization with momentum and adaptive learning rates
- Trains with fake quantization to minimize accuracy loss
- No dynamic memory
- 8-bit quantized weights significantly reduce model size
- Embedded friendly
- Adjustable weight and activation scaling factors

## Use Cases
- Embedded AI: Deploy neural networks on microcontrollers 
- Edge Computing: Low-power inference on resource-constrained devices
- Learning: Understanding neural network internals and quantization techniques
- Prototyping: Quick experimentation with small neural networks
- Real-time Applications: Fast inference due to integer-only operations

## Guide
1. Edit config.h to set your network architecture: 
- I (input layer size)
- H (hidden layers size)
- L (number of hidden layers)
- O (output layer size)
- TRAIN_FILE, INFER_FILE, MODEL_FILE
- activation, post processing, cost functions
- BATCH_SIZE, ALPHA and EPOCHS
2. Prepare data samples: The program expects I number of signed bytes for input followed by O number of signed bytes for output (This is by default but is configurable). See mnist_784_data_prep.py for example.
3. Train model: Compile and run train.c
4. Run inference: Compile and run infer.c

## Performance
### MNIST 784
- ~90% on MNIST test set
- Model size ~300KB (vs ~1200KB for float32)
- Infer time <1ms on modern CPUs
- Memory usage ~50KB during inference
### Impact
- 4x smaller model size compared to float32
- 2-4x faster inference on integer-optimized hardware
- Minimal accuracy loss (<1% vs full precision on MNIST)

## Examples
The examples/ directory contains models:
#### mnist_784
- mnist_784_train: 56,000 MNIST train samples in binary format
- mnist_784_infer: 14,000 MNIST test samples in binary format
- mnist_784_model: The trained model
- mnist_784_data_prep.py: mnist_784 csv to binary format converter
- config.h: The config for the mnist_784 model
