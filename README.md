# Quantized Model Training and Inference Kit

A minimal, dependency-free, allocation-agnostic stb-style library for quantized neural networks designed for embedded systems and resource constrained environments. 
This implementation uses 8-bit integer quantization for both weights and activations, enabling efficient inference on microcontrollers and edge devices.
It can achieve 4x smallermodel size, 2-4x faster inference, and minimal, if not none, accuracy loss.

## Features
- No dependencies
- INT8 weights and activations for maximum memory efficiency
- Easy to modify network topology via config.h
- Multiple activation, output processing and cost functions
- Optimization with momentum and adaptive learning rates
- Trains with fake quantization to minimize accuracy loss
- No dynamic memory (allocation-agnostic)
- 8-bit quantized weights significantly reduce model size
- Embedded friendly
- Adjustable weight and activation scaling factors

## Use Cases
- Embedded AI: Deploy neural networks on microcontrollers 
- Edge Computing: Low-power inference on resource-constrained devices
- Learning: Understanding neural network internals and quantization techniques
- Prototyping: Quick experimentation with small neural networks
- Real-time Applications: Fast inference due to integer-only operations

## Performance
### MNIST 784 (784-256-256-10)
- Accuracy: ~90%
- Model size: ~300KB (4x reduction from ~1.2MB for float32)
- Inference time: ~0.5 ms (~8 sec for 14000 inferences) on Intel Core i7-6500U 2.5 GHz
- Throughput: ~2000 inferences/second
- Training time: ~15 ms per sample on Intel Core i7-6500U 2.5 GHz
- Memory usage: ~400KB inference and ~4MB during training

## Examples
The examples/ directory contains models:
#### mnist_784
- mnist_784_train: 56,000 MNIST train samples in binary format
- mnist_784_infer: 14,000 MNIST test samples in binary format
- mnist_784_model: The trained model
- mnist_784_data_prep.py: mnist_784 csv to binary format converter
- qmtik_config.h: The config for the mnist_784 model
- train.c
- infer.c
- Makefile
