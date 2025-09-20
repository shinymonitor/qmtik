# Quantized Model Training and Inference Kit

A minimal, dependency-free, allocation-agnostic stb-style library for quantized neural networks designed for embedded systems and resource constrained environments. 
It uses uint8_t quantization for weights and activations to achieve 4x smaller model size, 2-4x faster inference, and minimal, if not none, accuracy loss.

On the MNIST dataset, QMTIK achieves ~95% test accuracy with a model that is just ~300KB and runs inference in ~0.5ms per sample on a modern CPU.

## Features
- uint8_t weights and activations for small memory usage and model size
- Adam optimization with batching
- Quantization-Aware Training to minimize accuracy loss
- Easy to modify network structure
- Multiple activation, output processing and cost functions
- Multiple learning rate decay functions
- Adjustable weight and activation scaling factors
- No dependencies
- No dynamic memory (allocation-agnostic)

## Use Cases
- Embedded AI: Deploy neural networks on microcontrollers 
- Edge Computing: Low-power inference on resource-constrained devices
- Learning: Understanding neural network internals and quantization techniques
- Prototyping: Quick experimentation with small neural networks
- Real-time Applications: Fast inference due to integer-only operations

## Performance
### MNIST 784 (784-256-256-10) [Benchmarked on an Intel Core i7-6500U @ 2.5 GHz]
- Accuracy: ~95%
- Model size: ~300KB (vs. ~1.2 MB for a float32 model)
- Infer time: ~0.5 ms per sample
- Train time: ~12 ms per sample
- Memory usage: ~400KB during inference, ~5MB during training

## Examples
**Copy the library header into the desired example directory and run the make file to build the training and inference binaries**
The examples/ directory contains models:
#### mnist_784
- mnist_784_train: 56,000 MNIST train samples in binary format
- mnist_784_infer: 14,000 MNIST test samples in binary format
- mnist_784_model: The trained model with ~95% accuracy
- mnist_784_data_prep.py: mnist_784 csv to binary format converter (refer this for the sample data file format)
- qmtik_config.h: The config for the mnist_784 model
- train.c
- infer.c
- Makefile
