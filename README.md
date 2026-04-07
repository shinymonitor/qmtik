<div align="center">
    <img src="assets/LOGO.png", width="200">
    <h1>QMTIK</h1>
</div>

QMTIK (Quantized Model Training and Inference Kit) minimal, dependency-free, allocation-agnostic stb-style library for quantized neural networks designed for embedded systems and resource constrained environments. 
It uses int8_t quantization for weights and activations to achieve 4x smaller model size, 4-16x faster inference, and minimal, if not none, accuracy loss.

On the MNIST 784 dataset, QMTIK achieves ~95% test accuracy with a model that is just ~300KB and runs inference in ~0.1ms per sample on a modern CPU.

## Features
- int8_t weights and activations for small memory usage and model size
- Adam optimization with batching
- Quantization-Aware Training to minimize accuracy loss
- Easy to modify network structure
- Multiple activation, output processing and cost functions
- Multiple learning rate decay functions
- Adjustable weight and activation scaling factors
- L2 regularization, gradient clipping and pruning
- No dependencies
- No dynamic memory (allocation-agnostic)

## Use Cases
- Embedded AI: Deploy neural networks on microcontrollers 
- Edge Computing: Low-power inference on resource-constrained devices
- Learning: Understanding neural network internals and quantization techniques
- Prototyping: Quick experimentation with small neural networks
- Real-time Applications: Fast inference due to integer-only operations

## Performance
QMTIK provides significant performance gains with minimal-to-zero accuracy loss. All benchmarks were run on an Intel Core i7-6500U @ 2.5 GHz.

|Dataset|Task|Accuracy|Model Size (FP32 -> INT8)|Inference Speedup|
|-------|----|--------|----------------|-----------------|
|MNIST|Digit Recognition|~95%|1.2 MB -> 327 KB **(4x smaller)**|**~14x faster**|
|Fashion-MNIST|Image Classification|~86%|1.2 MB -> 327 KB **(4x smaller)**|**~15x faster**|

**You can probably get even better accuracy with better hyperparameters**

## Examples
**Copy the library header into the desired example directory and run the make file to build the training and inference binaries**
The examples/ directory contains demos each with:
- ..._train: train samples in binary format
- ..._test: test samples in binary format
- ..._model: trained model
- make_sample_files.py: a non-portable model specific csv to sample file format converter (refer this for the sample data file format)
- qmtik_config.h: The config for the specific model
- train.c
- infer.c
- Makefile
