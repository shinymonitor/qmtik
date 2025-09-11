/*
QMTIK - Quantized Model Training and Inference Kit
A single-file header-only library for 8-bit quantized neural networks

USAGE:
    #define QMTIK_IMPLEMENTATION
    #include "qmtik.h"

CONFIGURATION:
    Before including, define these macros to configure your network:
    
    #define QMTIK_I 784        // Input size
    #define QMTIK_H 128        // Hidden layer size  
    #define QMTIK_L 2          // Number of hidden layers
    #define QMTIK_O 10         // Output size
    #define QMTIK_W_SCALE 0.01f // Weight quantization scale
    #define QMTIK_A_SCALE 1.0f  // Activation quantization scale
    
    // Optional training parameters
    #define QMTIK_EPOCHS 10
    #define QMTIK_ALPHA 0.001f
    #define QMTIK_BETA1 0.9f
    #define QMTIK_BETA2 0.999f
    #define QMTIK_EPS 1e-8f
    
    // Choose activation function (define one)
    #define QMTIK_RELU_ACTV
    // #define QMTIK_LEAKY_RELU_ACTV  
    // #define QMTIK_SIGMOID_ACTV
    // #define QMTIK_TANH_ACTV
    
    // Choose output processing (define one)
    #define QMTIK_LINEAR_PP
    // #define QMTIK_SOFT_MAX_PP
    // #define QMTIK_SIGMOID_PP
    
    // Choose cost function (define one) 
    #define QMTIK_MSE_COST
    // #define QMTIK_CROSS_ENTROPY_COST

    // Define debugging (optional)
    #define QMTIK_EPOCHS_DEBUG_UPDATE_POINT 1
    #define QMTIK_SAMPLE_NUMBER_DEBUG_UPDATE_POINT 1024
    #define QMTIK_TRAIN_DEBUG
    #define QMTIK_TEST_BEFORE_QUANT_DEBUG
    #define QMTIK_TEST_AFTER_QUANT_DEBUG

EXAMPLE:
Training:
    //=================NETWORK STRUCTURE===============
    #define QMTIK_I 784
    #define QMTIK_H 256
    #define QMTIK_L 2
    #define QMTIK_O 10
    #define QMTIK_W_SCALE 0.05f
    #define QMTIK_A_SCALE 0.5f
    //===ACTIVATION, POST PROCESSING, COST FUNCTIONS===
    #define QMTIK_LEAKY_RELU_ACTV
    #define QMTIK_SOFT_MAX_PP
    #define QMTIK_CROSS_ENTROPY_COST
    //=================TRAINING PARAMS=================
    #define QMTIK_ALPHA 0.001f
    #define QMTIK_EPOCHS 8
    #define QMTIK_BETA1 0.9f
    #define QMTIK_BETA2 0.999f
    #define QMTIK_EPS 1e-8f
    //=====================DEBUG=======================
    #define QMTIK_EPOCHS_DEBUG_UPDATE_POINT 1
    #define QMTIK_SAMPLE_NUMBER_DEBUG_UPDATE_POINT 1024
    #define QMTIK_TRAIN_DEBUG
    //=================================================
    #include "qmtik.h"
    
    int main() {
        QMTIK_Network network;
        QMTIK_init_weights(&network);
        
        FILE* train_file = fopen("train", "rb");
        QMTIK_train(&network, train_file);
        fclose(train_file);
        
        QMTIK_Model model;
        QMTIK_quantize_to_model(&network, &model);
        
        FILE* model_file=fopen("model", "wb");
        QMTIK_store_model(&model, model_file);
        fclose(model_file);

        return 0;
    }

Infering:
    //=================NETWORK STRUCTURE===============
    #define QMTIK_I 784
    #define QMTIK_H 256
    #define QMTIK_L 2
    #define QMTIK_O 10
    #define QMTIK_W_SCALE 0.05f
    #define QMTIK_A_SCALE 0.5f
    //===ACTIVATION, POST PROCESSING, COST FUNCTIONS===
    #define QMTIK_LEAKY_RELU_ACTV
    #define QMTIK_SOFT_MAX_PP
    #define QMTIK_CROSS_ENTROPY_COST
    //=====================DEBUG=======================
    #define QMTIK_EPOCHS_DEBUG_UPDATE_POINT 1
    #define QMTIK_SAMPLE_NUMBER_DEBUG_UPDATE_POINT 1024
    #define QMTIK_TEST_AFTER_QUANT_DEBUG
    //=================================================
    #include "qmtik.h"
    
    int main() {
        QMTIK_QNetwork q_network={0};

        FILE* q_model_file=fopen("mnist_784_model", "rb");
        QMTIK_load_model(&q_network, q_model_file);
        fclose(q_model_file);

        FILE* infer_file=fopen("mnist_784_infer", "rb");
        printf("PERFORMANCE AFTER QUANT: %f\n", QMTIK_test_after_quant(&q_network, infer_file));
        fclose(infer_file);

        return 0;
    }

MEMORY REQUIREMENTS:
    Training: ~sizeof(Network)
    Inference: ~sizeof(QNetwork)
    Model storage: ~sizeof(Model)
    
    This library is allocation-agnostic.
    Large networks may exceed default stack limits. Either increase stack limit or allocate on heap.

LICENSE:
    MIT License

    Copyright (c) 2025 Arin Upadhyay

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

VERSION HISTORY:
    1.0 (2025-09-11) Initial release
*/

#pragma once
#define QMTIK_VERSION "1.0"
//==================================================
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
//==================================================
#define QMTIK_MainT float
#define QMTIK_QWghtT int8_t
#define QMTIK_QActvT int8_t
#define QMTIK_QWghtT_MAX 127
#define QMTIK_QWghtT_MIN -128
#define QMTIK_QActvT_MAX 127
#define QMTIK_QActvT_MIN -128
#define QMTIK_LEAK 0.01f
#define QMTIK_CLAMP_MIN -88.0f
#define QMTIK_CLAMP_MAX 88.0f
//==================================================
typedef struct {QMTIK_MainT i_actv[QMTIK_I];} QMTIK_ILayer;
typedef struct {QMTIK_MainT ih_z[QMTIK_H]; QMTIK_MainT ih_wght[QMTIK_H][QMTIK_I], ih_bias[QMTIK_H];} QMTIK_IHLayer;
typedef struct {QMTIK_MainT hh_z[QMTIK_H]; QMTIK_MainT hh_wght[QMTIK_H][QMTIK_H], hh_bias[QMTIK_H];} QMTIK_HHLayer;
typedef struct {QMTIK_MainT o_z[QMTIK_O]; QMTIK_MainT o_wght[QMTIK_O][QMTIK_H], o_bias[QMTIK_O];} QMTIK_OLayer;
typedef struct {
    QMTIK_MainT m_ih_w[QMTIK_H][QMTIK_I], v_ih_w[QMTIK_H][QMTIK_I];
    QMTIK_MainT m_ih_b[QMTIK_H], v_ih_b[QMTIK_H];
    QMTIK_MainT m_hh_w[QMTIK_L][QMTIK_H][QMTIK_H], v_hh_w[QMTIK_L][QMTIK_H][QMTIK_H];
    QMTIK_MainT m_hh_b[QMTIK_L][QMTIK_H], v_hh_b[QMTIK_L][QMTIK_H];
    QMTIK_MainT m_o_w[QMTIK_O][QMTIK_H], v_o_w[QMTIK_O][QMTIK_H];
    QMTIK_MainT m_o_b[QMTIK_O], v_o_b[QMTIK_O];
    QMTIK_MainT dO[QMTIK_O], dHH[QMTIK_L][QMTIK_H], dIH[QMTIK_H];
    size_t t; QMTIK_MainT b1t, b2t;
} QMTIK_AdamState;
typedef struct {QMTIK_ILayer i_layer; QMTIK_IHLayer ih_layer; QMTIK_HHLayer hh_layers[QMTIK_L]; QMTIK_OLayer o_layer; QMTIK_AdamState adam_state;} QMTIK_Network;
typedef struct {QMTIK_QActvT input[QMTIK_I], output[QMTIK_O];} QMTIK_SamplePair;
typedef struct {QMTIK_QWghtT q_ih_wght[QMTIK_H][QMTIK_I], q_ih_bias[QMTIK_H], q_hh_wghts[QMTIK_L][QMTIK_H][QMTIK_H], q_hh_biases[QMTIK_L][QMTIK_H], q_o_wght[QMTIK_O][QMTIK_H], q_o_bias[QMTIK_O];} QMTIK_Model;
typedef struct {QMTIK_QActvT q_i_actv[QMTIK_I];} QMTIK_QILayer;
typedef struct {QMTIK_QActvT q_ih_actv[QMTIK_H]; QMTIK_QWghtT q_ih_wght[QMTIK_H][QMTIK_I], q_ih_bias[QMTIK_H];} QMTIK_QIHLayer;
typedef struct {QMTIK_QActvT q_hh_actv[QMTIK_H]; QMTIK_QWghtT q_hh_wght[QMTIK_H][QMTIK_H], q_hh_bias[QMTIK_H];} QMTIK_QHHLayer;
typedef struct {QMTIK_QActvT q_o_z[QMTIK_O]; QMTIK_QWghtT q_o_wght[QMTIK_O][QMTIK_H], q_o_bias[QMTIK_O];} QMTIK_QOLayer;
typedef struct {QMTIK_QILayer q_i_layer; QMTIK_QIHLayer q_ih_layer; QMTIK_QHHLayer q_hh_layers[QMTIK_L]; QMTIK_QOLayer q_o_layer;} QMTIK_QNetwork;
//==================================================
//==============USER VISIBLE FUNCTIONS==============
//==================================================
void QMTIK_init_weights(QMTIK_Network* network);

void QMTIK_train(QMTIK_Network* network, FILE* train_file);

void QMTIK_quantize_to_model(QMTIK_Network* network, QMTIK_Model* model);
void QMTIK_store_model(QMTIK_Model* model, FILE* q_model_file);
uint8_t QMTIK_load_model(QMTIK_QNetwork* q_network, FILE* q_model_file);

void QMTIK_infer_forward(QMTIK_QNetwork* q_network);

QMTIK_MainT QMTIK_test_before_quant(QMTIK_Network* network, FILE* test_file);
QMTIK_MainT QMTIK_test_after_quant(QMTIK_QNetwork* q_network, FILE* test_file);

void QMTIK_load_network_input(QMTIK_QNetwork* q_network, QMTIK_QActvT input[QMTIK_I]);
void QMTIK_get_network_output(QMTIK_QNetwork* q_network, QMTIK_QActvT output[QMTIK_O]);

size_t QMTIK_get_network_memory_usage(void);
size_t QMTIK_get_model_memory_usage(void);
size_t QMTIK_get_inference_memory_usage(void);
//==================================================
static inline QMTIK_MainT QMTIK_train_activation(QMTIK_MainT x);
static inline QMTIK_MainT QMTIK_train_activation_deriv(QMTIK_MainT x);
static inline QMTIK_QActvT QMTIK_infer_activation(QMTIK_MainT x);
static inline void QMTIK_train_post_process(QMTIK_MainT z[QMTIK_O]);
static inline void QMTIK_infer_post_process(QMTIK_QActvT z[QMTIK_O]);
static inline QMTIK_MainT QMTIK_train_cost(QMTIK_MainT output[QMTIK_O], QMTIK_QActvT expected[QMTIK_O]);
static inline QMTIK_MainT QMTIK_infer_cost(QMTIK_QActvT output[QMTIK_O], QMTIK_QActvT expected[QMTIK_O]);
//==================================================
static inline QMTIK_QWghtT QMTIK_quantize_w(QMTIK_MainT x);
static inline QMTIK_MainT QMTIK_fake_quantize_w(QMTIK_MainT x);
static inline QMTIK_QActvT QMTIK_quantize_a(QMTIK_MainT x);
static inline QMTIK_MainT QMTIK_fake_quantize_a(QMTIK_MainT x);
//==================================================
static inline uint8_t QMTIK_load_sample_pair(FILE* file, QMTIK_SamplePair* pair);
static inline void QMTIK_train_forward(QMTIK_Network* network);
static inline void QMTIK_train_step(QMTIK_Network* network, QMTIK_SamplePair sample_pair);
//==================================================
#ifdef QMTIK_IMPLEMENTATION
//==================================================
#ifdef QMTIK_RELU_ACTV
    static inline QMTIK_MainT QMTIK_train_activation(QMTIK_MainT x) {return x>0?x:0.0f;}
    static inline QMTIK_MainT QMTIK_train_activation_deriv(QMTIK_MainT x) {return x>0?1.0f:0.0f;}
    static inline QMTIK_QActvT QMTIK_infer_activation(QMTIK_MainT x) {return (QMTIK_QActvT)fmaxf(QMTIK_QActvT_MIN, fminf(QMTIK_QActvT_MAX, roundf(QMTIK_train_activation(x)/QMTIK_A_SCALE)));}
#endif
#ifdef QMTIK_LEAKY_RELU_ACTV
    static inline QMTIK_MainT QMTIK_train_activation(QMTIK_MainT x) {return x>0?x:QMTIK_LEAK*x;}
    static inline QMTIK_MainT QMTIK_train_activation_deriv(QMTIK_MainT x) {return x>0?1.0f:QMTIK_LEAK;}
    static inline QMTIK_QActvT QMTIK_infer_activation(QMTIK_MainT x) {return (QMTIK_QActvT)fmaxf(QMTIK_QActvT_MIN, fminf(QMTIK_QActvT_MAX, roundf(QMTIK_train_activation(x)/QMTIK_A_SCALE)));}
#endif
#ifdef QMTIK_SIGMOID_ACTV
    static inline QMTIK_MainT QMTIK_train_activation(QMTIK_MainT x) {return 1.0f/(1.0f+expf(-(fmaxf(QMTIK_CLAMP_MIN, fminf(QMTIK_CLAMP_MAX, x)))));}    
    static inline QMTIK_MainT QMTIK_train_activation_deriv(QMTIK_MainT x) {return QMTIK_train_activation(x)*(1.0f-QMTIK_train_activation(x));}
    static inline QMTIK_QActvT QMTIK_infer_activation(QMTIK_MainT x) {return (QMTIK_QActvT)fmaxf(QMTIK_QActvT_MIN, fminf(QMTIK_QActvT_MAX, roundf(QMTIK_train_activation(x)/QMTIK_A_SCALE)));}
#endif
#ifdef QMTIK_TANH_ACTV
    static inline QMTIK_MainT QMTIK_train_activation(QMTIK_MainT x) {return tanhf(fmaxf(QMTIK_CLAMP_MIN, fminf(QMTIK_CLAMP_MAX, x)));}
    static inline QMTIK_MainT QMTIK_train_activation_deriv(QMTIK_MainT x) {return 1.0f-QMTIK_train_activation(x)*QMTIK_train_activation(x);}
    static inline QMTIK_QActvT QMTIK_infer_activation(QMTIK_MainT x) {return (QMTIK_QActvT)fmaxf(QMTIK_QActvT_MIN, fminf(QMTIK_QActvT_MAX, roundf(QMTIK_train_activation(x)/QMTIK_A_SCALE)));}
#endif
//==================================================
#ifdef QMTIK_LINEAR_PP
    static inline void QMTIK_train_post_process(QMTIK_MainT z[QMTIK_O]) {for (size_t i=0; i<QMTIK_O; ++i) z[i]=fmaxf(-127.0f, fminf(127.0f, z[i]));}
    static inline void QMTIK_infer_post_process(QMTIK_QActvT z[QMTIK_O]) {(void)z;}
#endif
#ifdef QMTIK_SOFT_MAX_PP
    static inline void QMTIK_train_post_process(QMTIK_MainT z[QMTIK_O]) {
        QMTIK_MainT max_z=z[0];
        for (size_t i=1; i<QMTIK_O; ++i) if (z[i]>max_z) max_z=z[i];
        QMTIK_MainT sum=0.0f;
        for (size_t i=0; i<QMTIK_O; ++i) {z[i]=expf(z[i]-max_z); sum+=z[i];}
        for (size_t i=0; i<QMTIK_O; ++i) z[i]=(z[i]/sum)*127;
    }
    static inline void QMTIK_infer_post_process(QMTIK_QActvT z[QMTIK_O]) {
        float temp[QMTIK_O];
        float max_z=z[0]*QMTIK_A_SCALE;
        for (size_t i=1; i<QMTIK_O; ++i) if (z[i]*QMTIK_A_SCALE>max_z) max_z=z[i]*QMTIK_A_SCALE;
        float sum = 0.0f;
        for (size_t i=0; i<QMTIK_O; ++i) {temp[i]=expf(z[i]*QMTIK_A_SCALE-max_z); sum+=temp[i];}
        for (size_t i=0; i<QMTIK_O; ++i) {z[i]=(QMTIK_QActvT)roundf((temp[i]/sum)*127);}
    }
#endif
#ifdef QMTIK_SIGMOID_PP
    static inline void QMTIK_train_post_process(QMTIK_MainT z[QMTIK_O]) {for (size_t i=0; i<QMTIK_O; ++i){z[i]=fmaxf(QMTIK_CLAMP_MIN, fminf(QMTIK_CLAMP_MAX, z[i])); z[i]=(1.0f/(1.0f+expf(-z[i])))*127.0f;}}
    static inline void QMTIK_infer_post_process(QMTIK_QActvT z[QMTIK_O]) {for (size_t i=0; i<QMTIK_O; ++i) z[i]=(QMTIK_QActvT)roundf((1.0f/(1.0f+expf(-fmaxf(QMTIK_CLAMP_MIN, fminf(QMTIK_CLAMP_MAX, z[i]*QMTIK_A_SCALE)))))*127.0f);}
#endif
//==================================================
#ifdef QMTIK_MSE_COST
    static inline QMTIK_MainT QMTIK_train_cost(QMTIK_MainT output[QMTIK_O], QMTIK_QActvT expected[QMTIK_O]) {
        QMTIK_MainT total_error=0.0f;
        for (size_t i=0; i<QMTIK_O; ++i) {QMTIK_MainT diff=output[i]-(QMTIK_MainT)expected[i]; total_error+=diff*diff;}
        return total_error/QMTIK_O;
    }
    static inline QMTIK_MainT QMTIK_infer_cost(QMTIK_QActvT output[QMTIK_O], QMTIK_QActvT expected[QMTIK_O]) {
        QMTIK_MainT total_error=0.0f;
        for (size_t i = 0; i < QMTIK_O; ++i) {QMTIK_MainT diff=(QMTIK_MainT)output[i]-(QMTIK_MainT)expected[i]; total_error+=diff*diff;}
        return total_error/QMTIK_O;
    }
#endif
#ifdef QMTIK_CROSS_ENTROPY_COST
    static inline QMTIK_MainT QMTIK_train_cost(QMTIK_MainT output[QMTIK_O], QMTIK_QActvT expected[QMTIK_O]){
        int32_t pred_class=0;
        for(size_t i=1; i<QMTIK_O; ++i) if (output[i]>output[pred_class]) pred_class=i;
        int32_t exp_class=0;
        for(size_t i=1; i<QMTIK_O; ++i) if(expected[i]>expected[exp_class]) exp_class=i;
        return (pred_class==exp_class)?1:0;
    }
    static inline QMTIK_MainT QMTIK_infer_cost(QMTIK_QActvT output[QMTIK_O], QMTIK_QActvT expected[QMTIK_O]){
        int32_t pred_class=0;
        for(size_t i=1; i<QMTIK_O; ++i) if(output[i]>output[pred_class]) pred_class=i;
        int32_t exp_class=0;
        for(size_t i=1; i<QMTIK_O; ++i) if(expected[i]>expected[exp_class]) exp_class=i;
        return (pred_class==exp_class)?1:0;
    }
#endif
//==================================================
static inline uint8_t QMTIK_load_sample_pair(FILE* file, QMTIK_SamplePair* pair) {
    size_t r1=fread(pair->input, 1, QMTIK_I, file);
    size_t r2=fread(pair->output, 1, QMTIK_O, file);
    return (r1==QMTIK_I&&r2==QMTIK_O);
}
//==================================================
static inline QMTIK_QWghtT QMTIK_quantize_w(QMTIK_MainT x) {return (QMTIK_QWghtT)fmaxf(QMTIK_QWghtT_MIN, fminf(QMTIK_QWghtT_MAX, roundf(x/QMTIK_W_SCALE)));}
static inline QMTIK_MainT QMTIK_fake_quantize_w(QMTIK_MainT x) {return QMTIK_quantize_w(x)*QMTIK_W_SCALE;}
static inline QMTIK_QActvT QMTIK_quantize_a(QMTIK_MainT x) {return (QMTIK_QActvT)fmaxf(QMTIK_QActvT_MIN, fminf(QMTIK_QActvT_MAX, roundf(x/QMTIK_A_SCALE)));}
static inline QMTIK_MainT QMTIK_fake_quantize_a(QMTIK_MainT x) {return QMTIK_quantize_a(x)*QMTIK_A_SCALE;}
//==================================================
static inline void QMTIK_train_forward(QMTIK_Network* network) {
    QMTIK_MainT acc;
    for(size_t i=0; i<QMTIK_H; i++){
        acc=network->ih_layer.ih_bias[i];
        for(size_t j=0; j<QMTIK_I; ++j) acc+=QMTIK_fake_quantize_w(network->ih_layer.ih_wght[i][j])*QMTIK_fake_quantize_a(network->i_layer.i_actv[j]);
        network->ih_layer.ih_z[i]=acc;
    }
    for(size_t i=0; i<QMTIK_H; ++i){
        acc=network->hh_layers[0].hh_bias[i];
        for(size_t j=0; j<QMTIK_H; ++j) acc+=QMTIK_fake_quantize_w(network->hh_layers[0].hh_wght[i][j])*QMTIK_fake_quantize_a(QMTIK_train_activation(network->ih_layer.ih_z[j]));
        network->hh_layers[0].hh_z[i]=acc;
    }
    for(size_t l=1; l<QMTIK_L; ++l){
        for(size_t i=0; i<QMTIK_H; ++i){
            acc=network->hh_layers[l].hh_bias[i];
            for(size_t j=0; j<QMTIK_H; j++) acc+=QMTIK_fake_quantize_w(network->hh_layers[l].hh_wght[i][j])*QMTIK_fake_quantize_a(QMTIK_train_activation(network->hh_layers[l-1].hh_z[j]));
            network->hh_layers[l].hh_z[i]=acc;
        }
    }
    for(size_t i=0; i<QMTIK_O; ++i){
        acc=network->o_layer.o_bias[i];
        for(size_t j=0; j<QMTIK_H; ++j) acc+=QMTIK_fake_quantize_w(network->o_layer.o_wght[i][j])*QMTIK_fake_quantize_a(QMTIK_train_activation(network->hh_layers[QMTIK_L-1].hh_z[j]));
        network->o_layer.o_z[i]=acc;
    }
    QMTIK_train_post_process(network->o_layer.o_z);
}
void QMTIK_infer_forward(QMTIK_QNetwork* q_network) {
    QMTIK_MainT acc;
    for (size_t i=0; i<QMTIK_H; ++i){
        acc=q_network->q_ih_layer.q_ih_bias[i]*QMTIK_W_SCALE;
        for (size_t j=0; j<QMTIK_I; ++j) acc+=(q_network->q_ih_layer.q_ih_wght[i][j]*QMTIK_W_SCALE)*(q_network->q_i_layer.q_i_actv[j]*QMTIK_A_SCALE);
        q_network->q_ih_layer.q_ih_actv[i]=QMTIK_infer_activation(acc);
    }
    for (size_t i=0; i<QMTIK_H; ++i){
        acc=q_network->q_hh_layers[0].q_hh_bias[i]*QMTIK_W_SCALE;
        for (size_t j=0; j<QMTIK_H; ++j) acc+=(q_network->q_hh_layers[0].q_hh_wght[i][j]*QMTIK_W_SCALE)*(q_network->q_ih_layer.q_ih_actv[j]*QMTIK_A_SCALE);
        q_network->q_hh_layers[0].q_hh_actv[i]=QMTIK_infer_activation(acc);
    }
    for (size_t l=1; l<QMTIK_L; ++l){
        for (size_t i=0; i<QMTIK_H; ++i){
            acc=q_network->q_hh_layers[l].q_hh_bias[i]*QMTIK_W_SCALE;
            for (size_t j=0; j<QMTIK_H; ++j) acc+=(q_network->q_hh_layers[l].q_hh_wght[i][j]*QMTIK_W_SCALE)*(q_network->q_hh_layers[l-1].q_hh_actv[j]*QMTIK_A_SCALE);
            q_network->q_hh_layers[l].q_hh_actv[i]=QMTIK_infer_activation(acc);
        }
    }
    for (size_t i=0; i<QMTIK_O; ++i){
        acc=q_network->q_o_layer.q_o_bias[i]*QMTIK_W_SCALE;
        for (size_t j=0; j<QMTIK_H; ++j) acc+=(q_network->q_o_layer.q_o_wght[i][j]*QMTIK_W_SCALE)*(q_network->q_hh_layers[QMTIK_L-1].q_hh_actv[j]*QMTIK_A_SCALE);
        q_network->q_o_layer.q_o_z[i]=(QMTIK_QActvT)fmaxf(QMTIK_QActvT_MIN, fminf(QMTIK_QActvT_MAX, roundf(acc/QMTIK_A_SCALE)));
    }
    QMTIK_infer_post_process(q_network->q_o_layer.q_o_z);
}
//==================================================
static inline void QMTIK_train_step(QMTIK_Network* network, QMTIK_SamplePair sample_pair) {
    ++network->adam_state.t;
    network->adam_state.b1t*=QMTIK_BETA1;
    network->adam_state.b2t*=QMTIK_BETA2;
    for (size_t i=0; i<QMTIK_I; ++i) network->i_layer.i_actv[i]=sample_pair.input[i];
    QMTIK_train_forward(network);
    for (size_t i=0; i<QMTIK_O; i++) network->adam_state.dO[i]=network->o_layer.o_z[i]-(QMTIK_MainT)sample_pair.output[i];
    for (size_t i=0; i<QMTIK_H; ++i){
        QMTIK_MainT sum=0;
        for (size_t j=0; j<QMTIK_O; ++j) sum+=QMTIK_fake_quantize_w(network->o_layer.o_wght[j][i])*network->adam_state.dO[j];
        network->adam_state.dHH[QMTIK_L-1][i]=sum*QMTIK_train_activation_deriv(network->hh_layers[QMTIK_L-1].hh_z[i]);
    }
    for (int l=QMTIK_L-2; l>=0; --l){
        for (size_t i=0; i<QMTIK_H; ++i){
            QMTIK_MainT sum=0;
            for(size_t j=0; j<QMTIK_H; ++j) sum+=QMTIK_fake_quantize_w(network->hh_layers[l+1].hh_wght[j][i])*network->adam_state.dHH[l+1][j];
            network->adam_state.dHH[l][i]=sum*QMTIK_train_activation_deriv(network->hh_layers[l].hh_z[i]);
        }
    }
    for (size_t i=0; i<QMTIK_H; ++i){
        QMTIK_MainT sum=0;
        for (size_t j=0; j<QMTIK_H; ++j) sum+=QMTIK_fake_quantize_w(network->hh_layers[0].hh_wght[j][i])*network->adam_state.dHH[0][j];
        network->adam_state.dIH[i]=sum*QMTIK_train_activation_deriv(network->ih_layer.ih_z[i]);
    }
    for (size_t i=0; i<QMTIK_H; ++i){
        QMTIK_MainT dB=network->adam_state.dIH[i];
        network->adam_state.m_ih_b[i]=QMTIK_BETA1*network->adam_state.m_ih_b[i]+(1-QMTIK_BETA1)*dB;
        network->adam_state.v_ih_b[i]=QMTIK_BETA2*network->adam_state.v_ih_b[i]+(1-QMTIK_BETA2)*dB*dB;
        network->ih_layer.ih_bias[i]-=QMTIK_ALPHA*(network->adam_state.m_ih_b[i]/(1-network->adam_state.b1t))/(sqrtf(network->adam_state.v_ih_b[i]/(1-network->adam_state.b2t))+QMTIK_EPS);
        for (size_t j=0; j<QMTIK_I; ++j){
            QMTIK_MainT dW=network->adam_state.dIH[i]*QMTIK_fake_quantize_a(network->i_layer.i_actv[j]);
            network->adam_state.m_ih_w[i][j]=QMTIK_BETA1*network->adam_state.m_ih_w[i][j]+(1-QMTIK_BETA1)*dW;
            network->adam_state.v_ih_w[i][j]=QMTIK_BETA2*network->adam_state.v_ih_w[i][j]+(1-QMTIK_BETA2)*dW*dW;
            network->ih_layer.ih_wght[i][j]-=QMTIK_ALPHA*(network->adam_state.m_ih_w[i][j]/(1-network->adam_state.b1t))/(sqrtf(network->adam_state.v_ih_w[i][j]/(1-network->adam_state.b2t))+QMTIK_EPS);
        }
    }
    for (size_t l=0; l<QMTIK_L; ++l){
        for (size_t i=0; i<QMTIK_H; ++i){
            QMTIK_MainT dB=network->adam_state.dHH[l][i];
            network->adam_state.m_hh_b[l][i]=QMTIK_BETA1*network->adam_state.m_hh_b[l][i]+ (1-QMTIK_BETA1)*dB;
            network->adam_state.v_hh_b[l][i]=QMTIK_BETA2*network->adam_state.v_hh_b[l][i]+(1-QMTIK_BETA2)*dB*dB;
            network->hh_layers[l].hh_bias[i]-=QMTIK_ALPHA*(network->adam_state.m_hh_b[l][i]/(1-network->adam_state.b1t))/(sqrtf(network->adam_state.v_hh_b[l][i]/(1-network->adam_state.b2t))+QMTIK_EPS);
            for (size_t j=0; j<QMTIK_H; ++j){
                QMTIK_MainT dW=network->adam_state.dHH[l][i]*((l==0)?QMTIK_fake_quantize_a(QMTIK_train_activation(network->ih_layer.ih_z[j])):QMTIK_fake_quantize_a(QMTIK_train_activation(network->hh_layers[l-1].hh_z[j])));
                network->adam_state.m_hh_w[l][i][j]=QMTIK_BETA1*network->adam_state.m_hh_w[l][i][j]+(1-QMTIK_BETA1)*dW;
                network->adam_state.v_hh_w[l][i][j]=QMTIK_BETA2*network->adam_state.v_hh_w[l][i][j]+(1-QMTIK_BETA2)*dW*dW;
                network->hh_layers[l].hh_wght[i][j]-=QMTIK_ALPHA*(network->adam_state.m_hh_w[l][i][j]/(1-network->adam_state.b1t))/(sqrtf(network->adam_state.v_hh_w[l][i][j]/(1-network->adam_state.b2t))+QMTIK_EPS);
            }
        }
    }
    for (size_t i=0; i<QMTIK_O; ++i){
        QMTIK_MainT dB=network->adam_state.dO[i];
        network->adam_state.m_o_b[i]=QMTIK_BETA1*network->adam_state.m_o_b[i]+(1-QMTIK_BETA1)*dB;
        network->adam_state.v_o_b[i]=QMTIK_BETA2*network->adam_state.v_o_b[i]+(1-QMTIK_BETA2)*dB*dB;
        network->o_layer.o_bias[i]-=QMTIK_ALPHA*(network->adam_state.m_o_b[i]/(1-network->adam_state.b1t))/(sqrtf(network->adam_state.v_o_b[i]/(1-network->adam_state.b2t))+QMTIK_EPS);
        for (size_t j=0; j<QMTIK_H; ++j){
            QMTIK_MainT dW=network->adam_state.dO[i]*QMTIK_fake_quantize_a(QMTIK_train_activation(network->hh_layers[QMTIK_L-1].hh_z[j]));
            network->adam_state.m_o_w[i][j]=QMTIK_BETA1*network->adam_state.m_o_w[i][j]+(1-QMTIK_BETA1)*dW;
            network->adam_state.v_o_w[i][j]=QMTIK_BETA2*network->adam_state.v_o_w[i][j]+(1-QMTIK_BETA2)*dW*dW;
            network->o_layer.o_wght[i][j]-=QMTIK_ALPHA*(network->adam_state.m_o_w[i][j]/(1-network->adam_state.b1t))/(sqrtf(network->adam_state.v_o_w[i][j]/(1-network->adam_state.b2t))+QMTIK_EPS);
        }
    }
}
void QMTIK_train(QMTIK_Network* network, FILE* train_file){
    QMTIK_SamplePair sample;
    int _sample_number=0;
    #ifdef QMTIK_TRAIN_DEBUG
        printf("[QMTIK] ====TRAINING BEGIN====\n");
    #endif
    for (int _epoch=0; _epoch<QMTIK_EPOCHS; ++_epoch){
        #ifdef QMTIK_TRAIN_DEBUG
            if (_epoch%QMTIK_EPOCHS_DEBUG_UPDATE_POINT==0) printf("[QMTIK] EPOCH: %d\n", _epoch);
        #endif
        rewind(train_file);
        _sample_number=0;
        while (1){
            if (!QMTIK_load_sample_pair(train_file, &sample)) break;
            _sample_number++;
            #ifdef QMTIK_TRAIN_DEBUG
                if (_sample_number%QMTIK_SAMPLE_NUMBER_DEBUG_UPDATE_POINT==0) 
                    printf("[QMTIK] SAMPLE_NUMBER: %d\n", _sample_number);
            #endif
            QMTIK_train_step(network, sample);
        }
    }
}
//==================================================
void QMTIK_quantize_to_model(QMTIK_Network* network, QMTIK_Model* model){
    for (size_t i=0; i<QMTIK_H; ++i){
        model->q_ih_bias[i]=QMTIK_quantize_w(network->ih_layer.ih_bias[i]);
        for (size_t j=0; j<QMTIK_I; ++j) model->q_ih_wght[i][j]=QMTIK_quantize_w(network->ih_layer.ih_wght[i][j]);
    }
    for (size_t l=0; l<QMTIK_L; ++l){
        for (size_t i=0; i<QMTIK_H; ++i){
            model->q_hh_biases[l][i]=QMTIK_quantize_w(network->hh_layers[l].hh_bias[i]);
            for (size_t j=0; j<QMTIK_H; ++j) model->q_hh_wghts[l][i][j]=QMTIK_quantize_w(network->hh_layers[l].hh_wght[i][j]);
        }
    }
    for (size_t i=0; i<QMTIK_O; ++i){
        model->q_o_bias[i]=QMTIK_quantize_w(network->o_layer.o_bias[i]);
        for (size_t j=0; j<QMTIK_H; ++j) model->q_o_wght[i][j]=QMTIK_quantize_w(network->o_layer.o_wght[i][j]);
    }
}
void QMTIK_store_model(QMTIK_Model* model, FILE* q_model_file){fwrite(model, sizeof(QMTIK_Model), 1, q_model_file);}
uint8_t QMTIK_load_model(QMTIK_QNetwork* q_network, FILE* q_model_file) {
    QMTIK_Model model;
    if (!fread(&model, sizeof(QMTIK_Model), 1, q_model_file)){perror("[QMTIK] Failed to read model file"); return 1;}
    for (size_t i=0; i<QMTIK_H; ++i){
        q_network->q_ih_layer.q_ih_bias[i]=model.q_ih_bias[i];
        for (size_t j=0; j<QMTIK_I; ++j) q_network->q_ih_layer.q_ih_wght[i][j]=model.q_ih_wght[i][j];
    }
    for (size_t l=0; l<QMTIK_L; ++l){
        for (size_t i=0; i<QMTIK_H; ++i){
            q_network->q_hh_layers[l].q_hh_bias[i]=model.q_hh_biases[l][i];
            for (size_t j=0; j<QMTIK_H; ++j) q_network->q_hh_layers[l].q_hh_wght[i][j]=model.q_hh_wghts[l][i][j];
        }
    }
    for(size_t i=0; i<QMTIK_O; ++i){
        q_network->q_o_layer.q_o_bias[i]=model.q_o_bias[i];
        for(size_t j=0; j<QMTIK_H; ++j) q_network->q_o_layer.q_o_wght[i][j]=model.q_o_wght[i][j];
    }
    return 0;
}
//==================================================
void QMTIK_init_weights(QMTIK_Network* network){
    srand(time(NULL));
    for (size_t i=0; i<QMTIK_H; ++i){
        network->ih_layer.ih_bias[i]=0.0f;
        for (size_t j=0; j<QMTIK_I; ++j) network->ih_layer.ih_wght[i][j]=sqrtf(2.0f/(QMTIK_I+QMTIK_H))*((QMTIK_MainT)rand()/RAND_MAX-0.5f)*2.0f;
    }
    for (size_t l=0; l<QMTIK_L; ++l){
        for (size_t i=0; i<QMTIK_H; ++i) {
            network->hh_layers[l].hh_bias[i]=0.0f;
            for (size_t j=0; j<QMTIK_H; ++j) network->hh_layers[l].hh_wght[i][j]=sqrtf(2.0f/(QMTIK_H+QMTIK_H))*((QMTIK_MainT)rand()/RAND_MAX-0.5f)*2.0f;
        }
    }
    for (size_t i=0; i<QMTIK_O; ++i){
        network->o_layer.o_bias[i]=0.0f;
        for (size_t j=0; j<QMTIK_H; ++j) network->o_layer.o_wght[i][j]=sqrtf(2.0f/(QMTIK_H+QMTIK_O))*((QMTIK_MainT)rand()/RAND_MAX-0.5f)*2.0f;
    }
    memset(&network->adam_state, 0, sizeof(QMTIK_AdamState));
    network->adam_state.b1t = 1.0f;
    network->adam_state.b2t = 1.0f;
}
//==================================================
void QMTIK_load_network_input(QMTIK_QNetwork* q_network, QMTIK_QActvT input[QMTIK_I]) {for(size_t i=0; i<QMTIK_I; ++i) q_network->q_i_layer.q_i_actv[i]=input[i];}
void QMTIK_get_network_output(QMTIK_QNetwork* q_network, QMTIK_QActvT output[QMTIK_O]) {for(size_t i=0; i<QMTIK_O; ++i) output[i]=q_network->q_o_layer.q_o_z[i];}
//==================================================
QMTIK_MainT QMTIK_test_before_quant(QMTIK_Network* network, FILE* test_file){
    QMTIK_SamplePair pair;
    uint64_t total_cost=0;
    int _sample_number=0;
    rewind(test_file);
    #ifdef QMTIK_TEST_BEFORE_QUANT_DEBUG
        printf("[QMTIK] ====TEST BEFORE QUANT====\n");
    #endif
    while (1){
        if (!QMTIK_load_sample_pair(test_file, &pair)) break;
        for(size_t i=0; i<QMTIK_I; ++i) network->i_layer.i_actv[i]=(QMTIK_MainT)pair.input[i];
        QMTIK_train_forward(network);
        int32_t temp_cost=QMTIK_train_cost(network->o_layer.o_z, pair.output);
        #ifdef QMTIK_TEST_BEFORE_QUANT_DEBUG
            if (_sample_number%(QMTIK_SAMPLE_NUMBER_DEBUG_UPDATE_POINT)==0) {
                printf("[QMTIK] SAMPLE_NUMBER: %d\n", _sample_number);
                printf("[QMTIK] OUTPUT: ");
                for(size_t i=0; i<QMTIK_O; ++i) printf("%f,", network->o_layer.o_z[i]);
                printf("\n[QMTIK] EXPECTED: ");
                for(size_t i=0; i<QMTIK_O; ++i) printf("%d,", pair.output[i]);
                printf("\n[QMTIK] COST: %d\n", temp_cost);
            }
        #endif
        _sample_number+=1;
        total_cost+=temp_cost;
    }
    return (QMTIK_MainT)total_cost/_sample_number;
}
QMTIK_MainT QMTIK_test_after_quant(QMTIK_QNetwork* q_network, FILE* test_file){
    QMTIK_SamplePair pair;
    uint64_t total_cost=0;
    int _sample_number=0;
    rewind(test_file);
    #ifdef QMTIK_TEST_AFTER_QUANT_DEBUG
        printf("[QMTIK] ====TEST AFTER QUANT====\n");
    #endif
    while (1){
        if (!QMTIK_load_sample_pair(test_file, &pair)) break;
        QMTIK_load_network_input(q_network, pair.input);
        QMTIK_infer_forward(q_network);
        int32_t temp_cost=QMTIK_infer_cost(q_network->q_o_layer.q_o_z, pair.output);
        #ifdef QMTIK_TEST_AFTER_QUANT_DEBUG
            if (_sample_number%(QMTIK_SAMPLE_NUMBER_DEBUG_UPDATE_POINT)==0) {
                printf("[QMTIK] SAMPLE_NUMBER: %d\n", _sample_number);
                printf("[QMTIK] OUTPUT: ");
                for(size_t i=0; i<QMTIK_O; ++i) printf("%d,", q_network->q_o_layer.q_o_z[i]);
                printf("\n[QMTIK] EXPECTED: ");
                for(size_t i=0; i<QMTIK_O; ++i) printf("%d,", pair.output[i]);
                printf("\n[QMTIK] COST: %d\n", temp_cost);
            }
        #endif
        _sample_number+=1;
        total_cost+=temp_cost;
    }
    return (QMTIK_MainT)total_cost/_sample_number;
}
//==================================================
size_t QMTIK_get_network_memory_usage(void) {return sizeof(QMTIK_Network);}
size_t QMTIK_get_model_memory_usage(void) {return sizeof(QMTIK_Model);}
size_t QMTIK_get_inference_memory_usage(void) {return sizeof(QMTIK_QNetwork);}
//==================================================
#endif
