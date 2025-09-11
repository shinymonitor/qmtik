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
- ~90% on MNIST test set
- Model size ~300KB (vs ~1200KB for float32)
- Infer time ~0.5 ms (~8 sec for 14000 inferences) on Intel Core i7-6500U 2.5 GHz
- Train time ~15 ms per sample (~115 m for 56000 samples for 8 epochs) on Intel Core i7-6500U 2.5 GHz
- Memory usage only ~400KB during inference and only ~4MB during training
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
- qmtik_config.h: The config for the mnist_784 model
- train.c
- infer.c
- Makefile