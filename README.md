# Quantized Model Training and Inference Kit

A minimal, dependency-free, allocation-agnostic stb-style library for quantized neural networks designed for embedded systems and resource constrained environments. 
This implementation uses 8-bit integer quantization for both weights and activations.
It can achieve 4x smaller model size, 2-4x faster inference, and minimal, if not none, accuracy loss.

## Features
- No dependencies
- uint8_t weights and activations for small memory usage and model size
- Easy to modify network structure
- Multiple activation, output processing and cost functions
- Adam optimization
- Batching
- Multiple learning rate decay functions
- Trains with fake quantization to minimize accuracy loss
- No dynamic memory (allocation-agnostic)
- Adjustable weight and activation scaling factors

## Use Cases
- Embedded AI: Deploy neural networks on microcontrollers 
- Edge Computing: Low-power inference on resource-constrained devices
- Learning: Understanding neural network internals and quantization techniques
- Prototyping: Quick experimentation with small neural networks
- Real-time Applications: Fast inference due to integer-only operations

## Performance
### MNIST 784 (784-256-256-10)
- ~95% on MNIST test set
- Model size ~300KB (vs ~1200KB for float32)
- Infer time ~0.5 ms (~8 sec for 14000 inferences) on Intel Core i7-6500U 2.5 GHz
- Train time ~12 ms per sample (~90 m for 56000 samples for 8 epochs) on Intel Core i7-6500U 2.5 GHz
- Memory usage only ~400KB during inference and only ~5MB during training
### Impact
- 4x smaller model size compared to float32
- 2-4x faster inference (tested on x86_64 (Intel Core i7-6500U 2.5 GHz))
- Minimal accuracy loss (<1% compared to full precision)

## Examples
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
