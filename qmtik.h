/*
              ___________    
            _/  ____     \   
           | |_/    °__/° \  
          /  _/  /° / ___  | 
         | -/  _|__/\/_  ° | 
         | |  / \____  °  /  
          \__/\______°___/   
                   |___|     
                     \ \     
                      \_\    

                QMTIK

Quantized Model Training and Inference Kit
A single-file header-only library for 8-bit quantized neural networks

USAGE:
    #define QMTIK_IMPLEMENTATION  // STB style
    #define QMTIK_ENABLE_TRAINING  // Only needed if you want to train models, not just infer
    #include "qmtik.h"

CONFIGURATION:
    Before including, define these macros to configure your network:
    
    #define QMTIK_I 784        // Input size
    #define QMTIK_H 128        // Hidden layer size  
    #define QMTIK_L 2          // Number of hidden layers
    #define QMTIK_O 10         // Output size
    #define QMTIK_W_SCALE 0.25f // Weight quantization scale
    #define QMTIK_A_SCALE 0.02f  // Activation quantization scale

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
    // #define QMTIK_ARGMAX_COST
    
    // Rest are training parameters and are not needed for inference-only use
    // LR
    #define QMTIK_INIT_ALPHA 0.001f
    // Choose LR Decay function (define one)
    #define QMTIK_LR_NO_DECAY      // Constant learning rate
    // #define QMTIK_LR_STEP_DECAY
    // #define QMTIK_LR_EXPONENTIAL_DECAY  
    // #define QMTIK_LR_DECAY_RATE 0.95f
    // #define QMTIK_LR_DECAY_STEPS 1000

    #define QMTIK_L2_LAMBDA 0.0f
    #define QMTIK_GRADIENT_CLIP 1e13f
    #define QMTIK_PRUNE_THRESHOLD 0.0f

    #define QMTIK_BETA1 0.9f
    #define QMTIK_BETA2 0.999f
    #define QMTIK_EPS 1e-8f

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
    4.0 (2026-04-07) API cleanup, L2 regularization, gradient clipping and pruning
    3.0 (2025-09-26) Performance reports and simpler API
    2.0 (2025-09-20) LR Decay, Batching
    1.0 (2025-09-11) Initial release
*/

#ifndef QMTIK_H_
#define QMTIK_H_

#define QMTIK_VERSION "4.0"

#define QMTIK_DEF static inline
// ===============================================================
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
// ===============================================================
#ifndef QMTIK_B_Type
#define QMTIK_B_Type float
#endif
#ifndef QMTIK_Q_Type
#define QMTIK_Q_Type int8_t
#endif
#ifndef QMTIK_Q_Type_MAX
#define QMTIK_Q_Type_MAX INT8_MAX
#endif
#ifndef QMTIK_Q_Type_MIN
#define QMTIK_Q_Type_MIN INT8_MIN
#endif
// ===============================================================
typedef struct {QMTIK_B_Type i_actv[QMTIK_I];} QMTIK_B_ILayer;
typedef struct {QMTIK_B_Type ih_z[QMTIK_H]; QMTIK_B_Type ih_wght[QMTIK_H][QMTIK_I], ih_bias[QMTIK_H];} QMTIK_B_IHLayer;
typedef struct {QMTIK_B_Type hh_z[QMTIK_H]; QMTIK_B_Type hh_wght[QMTIK_H][QMTIK_H], hh_bias[QMTIK_H];} QMTIK_B_HHLayer;
typedef struct {QMTIK_B_Type o_z[QMTIK_O]; QMTIK_B_Type o_wght[QMTIK_O][QMTIK_H], o_bias[QMTIK_O];} QMTIK_B_OLayer;
typedef struct {QMTIK_Q_Type i_actv[QMTIK_I];} QMTIK_Q_ILayer;
typedef struct {QMTIK_Q_Type ih_actv[QMTIK_H]; QMTIK_Q_Type ih_wght[QMTIK_H][QMTIK_I], ih_bias[QMTIK_H];} QMTIK_Q_IHLayer;
typedef struct {QMTIK_Q_Type hh_actv[QMTIK_H]; QMTIK_Q_Type hh_wght[QMTIK_H][QMTIK_H], hh_bias[QMTIK_H];} QMTIK_Q_HHLayer;
typedef struct {QMTIK_Q_Type o_z[QMTIK_O]; QMTIK_Q_Type o_wght[QMTIK_O][QMTIK_H], o_bias[QMTIK_O];} QMTIK_Q_OLayer;
#ifdef QMTIK_ENABLE_TRAINING
    typedef struct {
        QMTIK_B_Type m_ih_w[QMTIK_H][QMTIK_I], v_ih_w[QMTIK_H][QMTIK_I];
        QMTIK_B_Type m_ih_b[QMTIK_H], v_ih_b[QMTIK_H];
        QMTIK_B_Type m_hh_w[QMTIK_L][QMTIK_H][QMTIK_H], v_hh_w[QMTIK_L][QMTIK_H][QMTIK_H];
        QMTIK_B_Type m_hh_b[QMTIK_L][QMTIK_H], v_hh_b[QMTIK_L][QMTIK_H];
        QMTIK_B_Type m_o_w[QMTIK_O][QMTIK_H], v_o_w[QMTIK_O][QMTIK_H];
        QMTIK_B_Type m_o_b[QMTIK_O], v_o_b[QMTIK_O];
        QMTIK_B_Type dO[QMTIK_O], dHH[QMTIK_L][QMTIK_H], dIH[QMTIK_H];
        QMTIK_B_Type acc_ih_w[QMTIK_H][QMTIK_I], acc_ih_b[QMTIK_H];
        QMTIK_B_Type acc_hh_w[QMTIK_L][QMTIK_H][QMTIK_H], acc_hh_b[QMTIK_L][QMTIK_H];
        QMTIK_B_Type acc_o_w[QMTIK_O][QMTIK_H], acc_o_b[QMTIK_O];
        size_t batch_count;
        QMTIK_B_Type current_alpha; size_t step_count;
        size_t t; QMTIK_B_Type b1t, b2t;
    } QMTIK_AdamState;
#endif
typedef struct {
    QMTIK_B_ILayer i_layer; 
    QMTIK_B_IHLayer ih_layer; 
    QMTIK_B_HHLayer hh_layers[QMTIK_L]; 
    QMTIK_B_OLayer o_layer; 
    #ifdef QMTIK_ENABLE_TRAINING
        QMTIK_AdamState adam_state;
    #endif
} QMTIK_B_Network;
typedef struct {
    QMTIK_Q_ILayer i_layer; 
    QMTIK_Q_IHLayer ih_layer; 
    QMTIK_Q_HHLayer hh_layers[QMTIK_L]; 
    QMTIK_Q_OLayer o_layer;
} QMTIK_Q_Network;
typedef struct {QMTIK_B_Type input[QMTIK_I], output[QMTIK_O];} QMTIK_B_Sample;
typedef struct {QMTIK_Q_Type input[QMTIK_I], output[QMTIK_O];} QMTIK_Q_Sample;
// ===============================================================
// PUBLIC API
// ===============================================================
QMTIK_DEF void QMTIK_B_load_input(QMTIK_B_Network* b_network, QMTIK_B_Type input[QMTIK_I]);
QMTIK_DEF void QMTIK_B_forward(QMTIK_B_Network* b_network);
QMTIK_DEF void QMTIK_B_get_output(QMTIK_B_Network* b_network, QMTIK_B_Type output[QMTIK_O]);

QMTIK_DEF void QMTIK_Q_load_input(QMTIK_Q_Network* q_network, QMTIK_Q_Type input[QMTIK_I]);
QMTIK_DEF void QMTIK_Q_forward(QMTIK_Q_Network* q_network);
QMTIK_DEF void QMTIK_Q_get_output(QMTIK_Q_Network* q_network, QMTIK_Q_Type output[QMTIK_O]);

#ifdef QMTIK_ENABLE_TRAINING
    QMTIK_DEF void QMTIK_B_init_weights(QMTIK_B_Network* b_network, uint32_t seed);
    QMTIK_DEF void QMTIK_B_accumulate_gradients(QMTIK_B_Network* b_network, QMTIK_B_Sample* sample);
    QMTIK_DEF void QMTIK_B_apply_gradients(QMTIK_B_Network* b_network);
    QMTIK_DEF void QMTIK_B_prune(QMTIK_B_Network* b_network);
#endif
QMTIK_DEF void QMTIK_B_quantize(QMTIK_B_Network* b_network, QMTIK_Q_Network* q_network);

QMTIK_DEF bool QMTIK_B_store_model_to_file(QMTIK_B_Network* b_network, FILE* b_model_file);
QMTIK_DEF bool QMTIK_B_load_model_from_file(FILE* b_model_file, QMTIK_B_Network* b_network);
QMTIK_DEF bool QMTIK_Q_store_model_to_file(QMTIK_Q_Network* q_network, FILE* q_model_file);
QMTIK_DEF bool QMTIK_Q_load_model_from_file(FILE* q_model_file, QMTIK_Q_Network* q_network);

QMTIK_DEF bool QMTIK_B_load_B_sample_from_file(FILE* b_sample_file, QMTIK_B_Sample* b_sample);
QMTIK_DEF bool QMTIK_B_load_Q_sample_from_file(FILE* q_sample_file, QMTIK_B_Sample* b_sample);
QMTIK_DEF bool QMTIK_Q_load_B_sample_from_file(FILE* b_sample_file, QMTIK_Q_Sample* q_sample);
QMTIK_DEF bool QMTIK_Q_load_Q_sample_from_file(FILE* q_sample_file, QMTIK_Q_Sample* q_sample);
// ===============================================================
// UTIL
// ===============================================================
QMTIK_DEF QMTIK_B_Type QMTIK_B_test_from_B_samples_file(QMTIK_B_Network* b_network, FILE* test_b_samples_file);
QMTIK_DEF QMTIK_B_Type QMTIK_Q_test_from_Q_samples_file(QMTIK_Q_Network* q_network, FILE* test_q_samples_file);
#ifdef QMTIK_ENABLE_TRAINING
    QMTIK_DEF void QMTIK_make_Q_model_to_file(
        QMTIK_B_Network* b_network, QMTIK_Q_Network* q_network, FILE* train_file, FILE* q_model_file, 
        size_t epochs, size_t batch_size,
        uint32_t seed, bool verbose
    );
#endif

// ===============================================================
// INTERNAL
// ===============================================================
QMTIK_DEF QMTIK_B_Type QMTIK_B_activation(QMTIK_B_Type x);
QMTIK_DEF QMTIK_B_Type QMTIK_B_activation_deriv(QMTIK_B_Type x);
QMTIK_DEF void QMTIK_B_post_process(QMTIK_B_Type z[QMTIK_O]);
QMTIK_DEF QMTIK_B_Type QMTIK_B_cost(QMTIK_B_Type output[QMTIK_O], QMTIK_B_Type expected[QMTIK_O]);
#ifdef QMTIK_ENABLE_TRAINING
    QMTIK_DEF void QMTIK_B_update_alpha(QMTIK_B_Type* current_alpha, size_t* step_count);
#endif

QMTIK_DEF QMTIK_Q_Type QMTIK_Q_activation(QMTIK_B_Type x);
QMTIK_DEF void QMTIK_Q_post_process(QMTIK_Q_Type z[QMTIK_O]);
QMTIK_DEF QMTIK_B_Type QMTIK_Q_cost(QMTIK_Q_Type output[QMTIK_O], QMTIK_Q_Type expected[QMTIK_O]);
// ===============================================================
QMTIK_DEF QMTIK_Q_Type QMTIK_quantize_a(QMTIK_B_Type x);
QMTIK_DEF QMTIK_B_Type QMTIK_fake_quantize_a(QMTIK_B_Type x);
QMTIK_DEF QMTIK_Q_Type QMTIK_quantize_w(QMTIK_B_Type x);
QMTIK_DEF QMTIK_B_Type QMTIK_fake_quantize_w(QMTIK_B_Type x);
// ===============================================================

#ifdef QMTIK_IMPLEMENTATION

#define QMTIK_LEAK 0.01f
#define QMTIK_CLAMP_MIN -88.0f
#define QMTIK_CLAMP_MAX 88.0f
#ifdef QMTIK_RELU_ACTV
    QMTIK_DEF QMTIK_B_Type QMTIK_B_activation(QMTIK_B_Type x) {
        return x > 0 ? x : 0.0f;
    }
    QMTIK_DEF QMTIK_B_Type QMTIK_B_activation_deriv(QMTIK_B_Type x) {
        return x > 0 ? 1.0f : 0.0f;
    }
    QMTIK_DEF QMTIK_Q_Type QMTIK_Q_activation(QMTIK_B_Type x) {
        return (QMTIK_Q_Type)fmaxf(QMTIK_Q_Type_MIN, fminf(QMTIK_Q_Type_MAX, roundf(QMTIK_B_activation(x) / QMTIK_A_SCALE)));
    }
#endif
#ifdef QMTIK_LEAKY_RELU_ACTV
    QMTIK_DEF QMTIK_B_Type QMTIK_B_activation(QMTIK_B_Type x) {
        return x > 0 ? x : QMTIK_LEAK * x;
    }
    QMTIK_DEF QMTIK_B_Type QMTIK_B_activation_deriv(QMTIK_B_Type x) {
        return x > 0 ? 1.0f : QMTIK_LEAK;
    }
    QMTIK_DEF QMTIK_Q_Type QMTIK_Q_activation(QMTIK_B_Type x) {
        return (QMTIK_Q_Type)fmaxf(QMTIK_Q_Type_MIN, fminf(QMTIK_Q_Type_MAX, roundf(QMTIK_B_activation(x) / QMTIK_A_SCALE)));
    }
#endif
#ifdef QMTIK_SIGMOID_ACTV
    QMTIK_DEF QMTIK_B_Type QMTIK_B_activation(QMTIK_B_Type x) {
        return 1.0f / (1.0f + expf(-(fmaxf(QMTIK_CLAMP_MIN, fminf(QMTIK_CLAMP_MAX, x)))));
    }
    QMTIK_DEF QMTIK_B_Type QMTIK_B_activation_deriv(QMTIK_B_Type x) {
        return QMTIK_B_activation(x) * (1.0f - QMTIK_B_activation(x));
    }
    QMTIK_DEF QMTIK_Q_Type QMTIK_Q_activation(QMTIK_B_Type x) {
        return (QMTIK_Q_Type)fmaxf(QMTIK_Q_Type_MIN, fminf(QMTIK_Q_Type_MAX, roundf(QMTIK_B_activation(x) / QMTIK_A_SCALE)));
    }
#endif
#ifdef QMTIK_TANH_ACTV
    QMTIK_DEF QMTIK_B_Type QMTIK_B_activation(QMTIK_B_Type x) {
        return tanhf(fmaxf(QMTIK_CLAMP_MIN, fminf(QMTIK_CLAMP_MAX, x)));
    }
    QMTIK_DEF QMTIK_B_Type QMTIK_B_activation_deriv(QMTIK_B_Type x) {
        return 1.0f - QMTIK_B_activation(x) * QMTIK_B_activation(x);
    }
    QMTIK_DEF QMTIK_Q_Type QMTIK_Q_activation(QMTIK_B_Type x) {
        return (QMTIK_Q_Type)fmaxf(QMTIK_Q_Type_MIN, fminf(QMTIK_Q_Type_MAX, roundf(QMTIK_B_activation(x) / QMTIK_A_SCALE)));
    }
#endif
// ===============================================================
#ifdef QMTIK_LINEAR_PP
    QMTIK_DEF void QMTIK_B_post_process(QMTIK_B_Type z[QMTIK_O]) {
        for (size_t i = 0; i < QMTIK_O; ++i) z[i] = fmaxf(QMTIK_Q_Type_MIN, fminf(QMTIK_Q_Type_MAX, z[i]));
    }
    QMTIK_DEF void QMTIK_Q_post_process(QMTIK_Q_Type z[QMTIK_O]) {
        (void)z;
    }
#endif
#ifdef QMTIK_SOFT_MAX_PP
    QMTIK_DEF void QMTIK_B_post_process(QMTIK_B_Type z[QMTIK_O]) {
        QMTIK_B_Type max_z = z[0];
        for (size_t i = 1; i < QMTIK_O; ++i) if (z[i] > max_z) max_z = z[i];
        QMTIK_B_Type sum = 0.0f;
        for (size_t i = 0; i < QMTIK_O; ++i) {
            z[i] = expf(z[i] - max_z); 
            sum += z[i];
        }
        for (size_t i = 0; i < QMTIK_O; ++i) z[i] = (z[i] / sum) * QMTIK_Q_Type_MAX;
    }
    QMTIK_DEF void QMTIK_Q_post_process(QMTIK_Q_Type z[QMTIK_O]) {
        float temp[QMTIK_O];
        float max_z = z[0] * QMTIK_A_SCALE;
        for (size_t i = 1; i < QMTIK_O; ++i) if (z[i] * QMTIK_A_SCALE > max_z) max_z = z[i] * QMTIK_A_SCALE;
        float sum = 0.0f;
        for (size_t i = 0; i < QMTIK_O; ++i) {
            temp[i] = expf(z[i] * QMTIK_A_SCALE - max_z);
            sum += temp[i];
        }
        for (size_t i = 0; i < QMTIK_O; ++i) z[i] = (QMTIK_Q_Type)roundf((temp[i] / sum) * QMTIK_Q_Type_MAX);
    }
#endif
#ifdef QMTIK_SIGMOID_PP
    QMTIK_DEF void QMTIK_B_post_process(QMTIK_B_Type z[QMTIK_O]) {
        for (size_t i = 0; i < QMTIK_O; ++i) z[i]=(1.0f / (1.0f + expf(-(fmaxf(QMTIK_CLAMP_MIN, fminf(QMTIK_CLAMP_MAX, z[i])))))) * QMTIK_Q_Type_MAX;
    }
    QMTIK_DEF void QMTIK_Q_post_process(QMTIK_Q_Type z[QMTIK_O]) {
        for (size_t i = 0; i < QMTIK_O; ++i) z[i] = (QMTIK_Q_Type)roundf((1.0f / (1.0f + expf(-fmaxf(QMTIK_CLAMP_MIN, fminf(QMTIK_CLAMP_MAX, z[i] * QMTIK_A_SCALE)))))*QMTIK_Q_Type_MAX);
    }
#endif
// ===============================================================
#ifdef QMTIK_MSE_COST
    QMTIK_DEF QMTIK_B_Type QMTIK_B_cost(QMTIK_B_Type output[QMTIK_O], QMTIK_B_Type expected[QMTIK_O]) {
        QMTIK_B_Type total_error = 0.0f;
        for (size_t i = 0; i < QMTIK_O; ++i) {
            QMTIK_B_Type diff = output[i] - (QMTIK_B_Type)expected[i]; 
            total_error += diff * diff;
        }
        return total_error / QMTIK_O;
    }
    QMTIK_DEF QMTIK_B_Type QMTIK_Q_cost(QMTIK_Q_Type output[QMTIK_O], QMTIK_Q_Type expected[QMTIK_O]) {
        QMTIK_B_Type total_error = 0.0f;
        for (size_t i = 0; i < QMTIK_O; ++i) {
            QMTIK_B_Type diff = output[i] - (QMTIK_B_Type)expected[i];
            total_error += diff * diff;
        }
        return total_error / QMTIK_O;
    }
#endif
#ifdef QMTIK_ARGMAX_COST
    QMTIK_DEF QMTIK_B_Type QMTIK_B_cost(QMTIK_B_Type output[QMTIK_O], QMTIK_B_Type expected[QMTIK_O]){
        int32_t pred_class = 0;
        for(size_t i = 1; i < QMTIK_O; ++i) if (output[i] > output[pred_class]) pred_class = i;
        int32_t exp_class = 0;
        for(size_t i = 1; i < QMTIK_O; ++i) if(expected[i] > expected[exp_class]) exp_class = i;
        return (pred_class == exp_class) ? 0 : 1;
    }
    QMTIK_DEF QMTIK_B_Type QMTIK_Q_cost(QMTIK_Q_Type output[QMTIK_O], QMTIK_Q_Type expected[QMTIK_O]){
        int32_t pred_class = 0;
        for(size_t i = 1; i < QMTIK_O; ++i) if (output[i] > output[pred_class]) pred_class = i;
        int32_t exp_class = 0;
        for(size_t i = 1; i < QMTIK_O; ++i) if(expected[i] > expected[exp_class]) exp_class = i;
        return (pred_class == exp_class) ? 0 : 1;
    }
#endif
// ===============================================================
#ifdef QMTIK_ENABLE_TRAINING
#ifdef QMTIK_LR_NO_DECAY
QMTIK_DEF void QMTIK_B_update_alpha(QMTIK_B_Type* current_alpha, size_t* step_count) {
    *current_alpha = QMTIK_INIT_ALPHA;
    *step_count += 1;
}
#endif
#ifdef QMTIK_LR_STEP_DECAY
QMTIK_DEF void QMTIK_B_update_alpha(QMTIK_B_Type* current_alpha, size_t* step_count) {
    *step_count += 1;
    if (*step_count % QMTIK_LR_DECAY_STEPS == 0) *current_alpha *= QMTIK_LR_DECAY_RATE;
}
#endif
#ifdef QMTIK_LR_EXPONENTIAL_DECAY
QMTIK_DEF void QMTIK_B_update_alpha(QMTIK_B_Type* current_alpha, size_t* step_count) {
    *step_count += 1;
    *current_alpha = QMTIK_INIT_ALPHA * powf(QMTIK_LR_DECAY_RATE, (QMTIK_B_Type)(*step_count) / QMTIK_LR_DECAY_STEPS);
}
#endif
#endif // QMTIK_ENABLE_TRAINING
// ===============================================================
QMTIK_DEF QMTIK_Q_Type QMTIK_quantize_a(QMTIK_B_Type x) {
    return (QMTIK_Q_Type)fmaxf(QMTIK_Q_Type_MIN, fminf(QMTIK_Q_Type_MAX, roundf(x/QMTIK_A_SCALE)));
}
QMTIK_DEF QMTIK_B_Type QMTIK_fake_quantize_a(QMTIK_B_Type x) {
    return QMTIK_quantize_a(x)*QMTIK_A_SCALE;
}
QMTIK_DEF QMTIK_Q_Type QMTIK_quantize_w(QMTIK_B_Type x) {
    return (QMTIK_Q_Type)fmaxf(QMTIK_Q_Type_MIN, fminf(QMTIK_Q_Type_MAX, roundf(x/QMTIK_W_SCALE)));
}
QMTIK_DEF QMTIK_B_Type QMTIK_fake_quantize_w(QMTIK_B_Type x) {
    return QMTIK_quantize_w(x)*QMTIK_W_SCALE;
}
// ===============================================================
QMTIK_DEF void QMTIK_B_load_input(QMTIK_B_Network* b_network, QMTIK_B_Type input[QMTIK_I]) {
    memcpy(b_network->i_layer.i_actv, input, sizeof(QMTIK_B_Type) * QMTIK_I);
}
QMTIK_DEF void QMTIK_B_forward(QMTIK_B_Network* b_network) {
    QMTIK_B_Type acc;
    for(size_t i = 0; i < QMTIK_H; ++i){
        acc = QMTIK_fake_quantize_w(b_network->ih_layer.ih_bias[i]);
        for(size_t j = 0; j < QMTIK_I; ++j) acc += QMTIK_fake_quantize_w(b_network->ih_layer.ih_wght[i][j]) * QMTIK_fake_quantize_a(b_network->i_layer.i_actv[j]);
        b_network->ih_layer.ih_z[i] = acc;
    }
    for(size_t i = 0; i < QMTIK_H; ++i){
        acc = QMTIK_fake_quantize_w(b_network->hh_layers[0].hh_bias[i]);
        for(size_t j = 0; j < QMTIK_H; ++j) acc += QMTIK_fake_quantize_w(b_network->hh_layers[0].hh_wght[i][j]) * QMTIK_fake_quantize_a(QMTIK_B_activation(b_network->ih_layer.ih_z[j]));
        b_network->hh_layers[0].hh_z[i] = acc;
    }
    for(size_t l = 1; l < QMTIK_L; ++l){
        for(size_t i = 0; i < QMTIK_H; ++i){
            acc = QMTIK_fake_quantize_w(b_network->hh_layers[l].hh_bias[i]);
            for(size_t j = 0; j < QMTIK_H; ++j) acc += QMTIK_fake_quantize_w(b_network->hh_layers[l].hh_wght[i][j]) * QMTIK_fake_quantize_a(QMTIK_B_activation(b_network->hh_layers[l - 1].hh_z[j]));
            b_network->hh_layers[l].hh_z[i] = acc;
        }
    }
    for(size_t i = 0; i < QMTIK_O; ++i){
        acc = QMTIK_fake_quantize_w(b_network->o_layer.o_bias[i]);
        for(size_t j = 0; j < QMTIK_H; ++j) acc += QMTIK_fake_quantize_w(b_network->o_layer.o_wght[i][j]) * QMTIK_fake_quantize_a(QMTIK_B_activation(b_network->hh_layers[QMTIK_L - 1].hh_z[j]));
        b_network->o_layer.o_z[i] = acc;
    }
    QMTIK_B_post_process(b_network->o_layer.o_z);
}
QMTIK_DEF void QMTIK_B_get_output(QMTIK_B_Network* b_network, QMTIK_B_Type output[QMTIK_O]){
    memcpy(output, b_network->o_layer.o_z, sizeof(QMTIK_B_Type) * QMTIK_O);
}

QMTIK_DEF void QMTIK_Q_load_input(QMTIK_Q_Network* q_network, QMTIK_Q_Type input[QMTIK_I]) {
    memcpy(q_network->i_layer.i_actv, input, sizeof(QMTIK_Q_Type) * QMTIK_I);
}
QMTIK_DEF void QMTIK_Q_forward(QMTIK_Q_Network* q_network) {
    QMTIK_B_Type acc;
    for(size_t i = 0; i < QMTIK_H; ++i){
        acc = q_network->ih_layer.ih_bias[i] * QMTIK_W_SCALE;
        for(size_t j = 0; j < QMTIK_I; ++j) acc += (q_network->ih_layer.ih_wght[i][j] * QMTIK_W_SCALE) * (q_network->i_layer.i_actv[j] * QMTIK_A_SCALE);
        q_network->ih_layer.ih_actv[i] = QMTIK_Q_activation(acc);
    }
    for (size_t i = 0; i < QMTIK_H; ++i){
        acc = q_network->hh_layers[0].hh_bias[i] * QMTIK_W_SCALE;
        for (size_t j = 0; j < QMTIK_H; ++j) acc += (q_network->hh_layers[0].hh_wght[i][j] * QMTIK_W_SCALE) * (q_network->ih_layer.ih_actv[j] * QMTIK_A_SCALE);
        q_network->hh_layers[0].hh_actv[i] = QMTIK_Q_activation(acc);
    }
    for (size_t l = 1; l < QMTIK_L; ++l){
        for (size_t i = 0; i < QMTIK_H; ++i){
            acc = q_network->hh_layers[l].hh_bias[i] * QMTIK_W_SCALE;
            for (size_t j = 0; j < QMTIK_H; ++j) acc += (q_network->hh_layers[l].hh_wght[i][j] * QMTIK_W_SCALE) * (q_network->hh_layers[l - 1].hh_actv[j] * QMTIK_A_SCALE);
            q_network->hh_layers[l].hh_actv[i] = QMTIK_Q_activation(acc);
        }
    }
    for (size_t i = 0; i < QMTIK_O; ++i){
        acc = q_network->o_layer.o_bias[i] * QMTIK_W_SCALE;
        for (size_t j = 0; j < QMTIK_H; ++j) acc += (q_network->o_layer.o_wght[i][j] * QMTIK_W_SCALE) * (q_network->hh_layers[QMTIK_L - 1].hh_actv[j] * QMTIK_A_SCALE);
        q_network->o_layer.o_z[i] = QMTIK_quantize_a(acc);
    }
    QMTIK_Q_post_process(q_network->o_layer.o_z);
}
QMTIK_DEF void QMTIK_Q_get_output(QMTIK_Q_Network* q_network, QMTIK_Q_Type output[QMTIK_O]){
    memcpy(output, q_network->o_layer.o_z, sizeof(QMTIK_Q_Type) * QMTIK_O);
}
// ===============================================================
#ifdef QMTIK_ENABLE_TRAINING
QMTIK_DEF void QMTIK_B_init_weights(QMTIK_B_Network* b_network, uint32_t seed) {
    srand(seed);
    for (size_t i = 0; i < QMTIK_H; ++i){
        b_network->ih_layer.ih_bias[i] = 0.0f;
        for (size_t j = 0; j < QMTIK_I; ++j) b_network->ih_layer.ih_wght[i][j] = sqrtf(2.0f / (QMTIK_I + QMTIK_H)) * ((QMTIK_B_Type)rand() / RAND_MAX - 0.5f) * 2.0f;
    }
    for (size_t l = 0; l < QMTIK_L; ++l){
        for (size_t i = 0; i < QMTIK_H; ++i) {
            b_network->hh_layers[l].hh_bias[i] = 0.0f;
            for (size_t j = 0; j < QMTIK_H; ++j) b_network->hh_layers[l].hh_wght[i][j] = sqrtf(2.0f / (QMTIK_H + QMTIK_H)) * ((QMTIK_B_Type)rand() / RAND_MAX - 0.5f) * 2.0f;
        }
    }
    for (size_t i = 0; i < QMTIK_O; ++i){
        b_network->o_layer.o_bias[i] = 0.0f;
        for (size_t j = 0; j < QMTIK_H; ++j) b_network->o_layer.o_wght[i][j] = sqrtf(2.0f / (QMTIK_H + QMTIK_O)) * ((QMTIK_B_Type)rand() / RAND_MAX - 0.5f) * 2.0f;
    }
    memset(&b_network->adam_state, 0, sizeof(QMTIK_AdamState));
    b_network->adam_state.current_alpha = QMTIK_INIT_ALPHA;
    b_network->adam_state.b1t = 1.0f;
    b_network->adam_state.b2t = 1.0f;
}
QMTIK_DEF void QMTIK_B_accumulate_gradients(QMTIK_B_Network* b_network, QMTIK_B_Sample* b_sample) {
    for(size_t i = 0; i < QMTIK_I; ++i) b_network->i_layer.i_actv[i] = b_sample->input[i];
    QMTIK_B_forward(b_network);
    for (size_t i = 0; i < QMTIK_O; ++i) b_network->adam_state.dO[i] = b_network->o_layer.o_z[i] - b_sample->output[i];
    for (size_t i = 0; i < QMTIK_H; ++i){
        QMTIK_B_Type sum = 0;
        for (size_t j = 0; j < QMTIK_O; ++j) sum += QMTIK_fake_quantize_w(b_network->o_layer.o_wght[j][i]) * b_network->adam_state.dO[j];
        b_network->adam_state.dHH[QMTIK_L - 1][i] = sum*QMTIK_B_activation_deriv(b_network->hh_layers[QMTIK_L - 1].hh_z[i]);
    }
    for (int l = QMTIK_L - 2; l >= 0; --l){
        for (size_t i = 0; i < QMTIK_H; ++i){
            QMTIK_B_Type sum = 0;
            for(size_t j = 0; j < QMTIK_H; ++j) sum += QMTIK_fake_quantize_w(b_network->hh_layers[l + 1].hh_wght[j][i]) * b_network->adam_state.dHH[l + 1][j];
            b_network->adam_state.dHH[l][i] = sum * QMTIK_B_activation_deriv(b_network->hh_layers[l].hh_z[i]);
        }
    }
    for (size_t i = 0; i < QMTIK_H; ++i){
        QMTIK_B_Type sum = 0;
        for (size_t j = 0; j < QMTIK_H; ++j) sum += QMTIK_fake_quantize_w(b_network->hh_layers[0].hh_wght[j][i]) * b_network->adam_state.dHH[0][j];
        b_network->adam_state.dIH[i] = sum * QMTIK_B_activation_deriv(b_network->ih_layer.ih_z[i]);
    }
    for (size_t i = 0; i < QMTIK_H; ++i){
        b_network->adam_state.acc_ih_b[i]+=b_network->adam_state.dIH[i];
        for (size_t j = 0; j < QMTIK_I; ++j) b_network->adam_state.acc_ih_w[i][j] += b_network->adam_state.dIH[i] * QMTIK_fake_quantize_a(b_network->i_layer.i_actv[j]);
    }
    for (size_t l = 0; l < QMTIK_L; ++l){
        for (size_t i = 0; i < QMTIK_H; ++i){
            b_network->adam_state.acc_hh_b[l][i] += b_network->adam_state.dHH[l][i];
            for (size_t j = 0; j < QMTIK_H; ++j){
                QMTIK_B_Type prev_actv = (l == 0) ? QMTIK_fake_quantize_a(QMTIK_B_activation(b_network->ih_layer.ih_z[j])) : QMTIK_fake_quantize_a(QMTIK_B_activation(b_network->hh_layers[l - 1].hh_z[j]));
                b_network->adam_state.acc_hh_w[l][i][j] += b_network->adam_state.dHH[l][i] * prev_actv;
            }
        }
    }
    for (size_t i = 0; i < QMTIK_O; ++i){
        b_network->adam_state.acc_o_b[i] += b_network->adam_state.dO[i];
        for (size_t j = 0; j < QMTIK_H; ++j) b_network->adam_state.acc_o_w[i][j] += b_network->adam_state.dO[i] * QMTIK_fake_quantize_a(QMTIK_B_activation(b_network->hh_layers[QMTIK_L - 1].hh_z[j]));
    }
    b_network->adam_state.batch_count++;
}

QMTIK_DEF void QMTIK_B_apply_gradients(QMTIK_B_Network* b_network) {
    QMTIK_B_update_alpha(&b_network->adam_state.current_alpha, &b_network->adam_state.step_count);
    ++b_network->adam_state.t;
    b_network->adam_state.b1t *= QMTIK_BETA1;
    b_network->adam_state.b2t *= QMTIK_BETA2;
    if (b_network->adam_state.batch_count == 0) return;
    for (size_t i = 0; i < QMTIK_H; ++i){
        QMTIK_B_Type avg_dB = b_network->adam_state.acc_ih_b[i] / b_network->adam_state.batch_count;
        avg_dB = fmaxf(-QMTIK_GRADIENT_CLIP, fminf(QMTIK_GRADIENT_CLIP, avg_dB));
        b_network->adam_state.m_ih_b[i] = QMTIK_BETA1 * b_network->adam_state.m_ih_b[i] + (1 - QMTIK_BETA1) * avg_dB;
        b_network->adam_state.v_ih_b[i] = QMTIK_BETA2 * b_network->adam_state.v_ih_b[i] + (1 - QMTIK_BETA2) * avg_dB * avg_dB;
        b_network->ih_layer.ih_bias[i] -= (b_network->adam_state.current_alpha) * (b_network->adam_state.m_ih_b[i] / (1-b_network->adam_state.b1t)) / (sqrtf(b_network->adam_state.v_ih_b[i] / (1 - b_network->adam_state.b2t)) + QMTIK_EPS);
        for (size_t j = 0; j < QMTIK_I; ++j){
            QMTIK_B_Type avg_dW = b_network->adam_state.acc_ih_w[i][j] / b_network->adam_state.batch_count;
            avg_dW = fmaxf(-QMTIK_GRADIENT_CLIP, fminf(QMTIK_GRADIENT_CLIP, avg_dW));
            b_network->adam_state.m_ih_w[i][j] = QMTIK_BETA1 * b_network->adam_state.m_ih_w[i][j] + (1 - QMTIK_BETA1) * avg_dW;
            b_network->adam_state.v_ih_w[i][j] = QMTIK_BETA2 * b_network->adam_state.v_ih_w[i][j] + (1 - QMTIK_BETA2) * avg_dW * avg_dW;
            b_network->ih_layer.ih_wght[i][j] -= (b_network->adam_state.current_alpha) * (b_network->adam_state.m_ih_w[i][j] / (1 - b_network->adam_state.b1t)) / (sqrtf(b_network->adam_state.v_ih_w[i][j] / (1 - b_network->adam_state.b2t)) + QMTIK_EPS);
            b_network->ih_layer.ih_wght[i][j] -= QMTIK_L2_LAMBDA * b_network->ih_layer.ih_wght[i][j];
        }
    }
    for (size_t l=0; l<QMTIK_L; ++l){
        for (size_t i=0; i<QMTIK_H; ++i){
            QMTIK_B_Type avg_dB = b_network->adam_state.acc_hh_b[l][i] / b_network->adam_state.batch_count;
            avg_dB = fmaxf(-QMTIK_GRADIENT_CLIP, fminf(QMTIK_GRADIENT_CLIP, avg_dB));
            b_network->adam_state.m_hh_b[l][i] = QMTIK_BETA1 * b_network->adam_state.m_hh_b[l][i] + (1 - QMTIK_BETA1) * avg_dB;
            b_network->adam_state.v_hh_b[l][i] = QMTIK_BETA2 * b_network->adam_state.v_hh_b[l][i] + (1 - QMTIK_BETA2) * avg_dB * avg_dB;
            b_network->hh_layers[l].hh_bias[i] -= (b_network->adam_state.current_alpha) * (b_network->adam_state.m_hh_b[l][i] / (1 - b_network->adam_state.b1t)) / (sqrtf(b_network->adam_state.v_hh_b[l][i] / (1 - b_network->adam_state.b2t)) + QMTIK_EPS);
            for (size_t j = 0; j < QMTIK_H; ++j){
                QMTIK_B_Type avg_dW = b_network->adam_state.acc_hh_w[l][i][j] / b_network->adam_state.batch_count;
                avg_dW = fmaxf(-QMTIK_GRADIENT_CLIP, fminf(QMTIK_GRADIENT_CLIP, avg_dW));
                b_network->adam_state.m_hh_w[l][i][j] = QMTIK_BETA1 * b_network->adam_state.m_hh_w[l][i][j] + (1 - QMTIK_BETA1) * avg_dW;
                b_network->adam_state.v_hh_w[l][i][j] = QMTIK_BETA2 * b_network->adam_state.v_hh_w[l][i][j] + (1 - QMTIK_BETA2) * avg_dW * avg_dW;
                b_network->hh_layers[l].hh_wght[i][j] -= (b_network->adam_state.current_alpha) * (b_network->adam_state.m_hh_w[l][i][j] / (1 - b_network->adam_state.b1t)) / (sqrtf(b_network->adam_state.v_hh_w[l][i][j] / (1 - b_network->adam_state.b2t)) + QMTIK_EPS);
                b_network->hh_layers[l].hh_wght[i][j] -= QMTIK_L2_LAMBDA * b_network->hh_layers[l].hh_wght[i][j];
            }
        }
    }
    for (size_t i=0; i<QMTIK_O; ++i){
        QMTIK_B_Type avg_dB = b_network->adam_state.acc_o_b[i] / b_network->adam_state.batch_count;
        avg_dB = fmaxf(-QMTIK_GRADIENT_CLIP, fminf(QMTIK_GRADIENT_CLIP, avg_dB));
        b_network->adam_state.m_o_b[i] = QMTIK_BETA1 * b_network->adam_state.m_o_b[i] + (1 - QMTIK_BETA1) * avg_dB;
        b_network->adam_state.v_o_b[i] = QMTIK_BETA2 * b_network->adam_state.v_o_b[i] + (1 - QMTIK_BETA2) * avg_dB * avg_dB;
        b_network->o_layer.o_bias[i] -= (b_network->adam_state.current_alpha) * (b_network->adam_state.m_o_b[i] / (1 - b_network->adam_state.b1t)) / (sqrtf(b_network->adam_state.v_o_b[i] / (1 - b_network->adam_state.b2t)) + QMTIK_EPS);
        for (size_t j=0; j<QMTIK_H; ++j){
            QMTIK_B_Type avg_dW = b_network->adam_state.acc_o_w[i][j] / b_network->adam_state.batch_count;
            avg_dW = fmaxf(-QMTIK_GRADIENT_CLIP, fminf(QMTIK_GRADIENT_CLIP, avg_dW));
            b_network->adam_state.m_o_w[i][j] = QMTIK_BETA1 * b_network->adam_state.m_o_w[i][j] + (1 - QMTIK_BETA1) * avg_dW;
            b_network->adam_state.v_o_w[i][j] = QMTIK_BETA2 * b_network->adam_state.v_o_w[i][j] + (1 - QMTIK_BETA2) * avg_dW * avg_dW;
            b_network->o_layer.o_wght[i][j] -= (b_network->adam_state.current_alpha) * (b_network->adam_state.m_o_w[i][j] / (1 - b_network->adam_state.b1t)) / (sqrtf(b_network->adam_state.v_o_w[i][j] / (1 - b_network->adam_state.b2t)) + QMTIK_EPS);
            b_network->o_layer.o_wght[i][j] -= QMTIK_L2_LAMBDA * b_network->o_layer.o_wght[i][j];
        }
    }
    memset(b_network->adam_state.acc_ih_w, 0, sizeof(b_network->adam_state.acc_ih_w));
    memset(b_network->adam_state.acc_ih_b, 0, sizeof(b_network->adam_state.acc_ih_b));
    memset(b_network->adam_state.acc_hh_w, 0, sizeof(b_network->adam_state.acc_hh_w));
    memset(b_network->adam_state.acc_hh_b, 0, sizeof(b_network->adam_state.acc_hh_b));
    memset(b_network->adam_state.acc_o_w, 0, sizeof(b_network->adam_state.acc_o_w));
    memset(b_network->adam_state.acc_o_b, 0, sizeof(b_network->adam_state.acc_o_b));
    b_network->adam_state.batch_count = 0;
}
QMTIK_DEF void QMTIK_B_prune(QMTIK_B_Network* b_network) {
    for (size_t i = 0; i < QMTIK_H; ++i){
        if (fabsf(b_network->ih_layer.ih_bias[i]) < QMTIK_PRUNE_THRESHOLD) b_network->ih_layer.ih_bias[i] = 0.0f;
        for (size_t j = 0; j < QMTIK_I; ++j) if (fabsf(b_network->ih_layer.ih_wght[i][j]) < QMTIK_PRUNE_THRESHOLD) b_network->ih_layer.ih_wght[i][j] = 0.0f;
    }
    for (size_t l = 0; l < QMTIK_L; ++l){
        for (size_t i = 0; i < QMTIK_H; ++i){
            if (fabsf(b_network->hh_layers[l].hh_bias[i]) < QMTIK_PRUNE_THRESHOLD) b_network->hh_layers[l].hh_bias[i] = 0.0f;
            for (size_t j = 0; j < QMTIK_H; ++j) if (fabsf(b_network->hh_layers[l].hh_wght[i][j]) < QMTIK_PRUNE_THRESHOLD) b_network->hh_layers[l].hh_wght[i][j] = 0.0f;
        }
    }
    for (size_t i = 0; i < QMTIK_O; ++i){
        if (fabsf(b_network->o_layer.o_bias[i]) < QMTIK_PRUNE_THRESHOLD) b_network->o_layer.o_bias[i] = 0.0f;
        for (size_t j = 0; j < QMTIK_H; ++j) if (fabsf(b_network->o_layer.o_wght[i][j]) < QMTIK_PRUNE_THRESHOLD) b_network->o_layer.o_wght[i][j] = 0.0f;
    }
}
#endif // QMTIK_ENABLE_TRAINING
// ===============================================================
QMTIK_DEF void QMTIK_B_quantize(QMTIK_B_Network* b_network, QMTIK_Q_Network* q_network) {
    for (size_t i = 0; i < QMTIK_H; ++i){
        q_network->ih_layer.ih_bias[i] = QMTIK_quantize_w(b_network->ih_layer.ih_bias[i]);
        for (size_t j = 0; j < QMTIK_I; ++j) q_network->ih_layer.ih_wght[i][j] = QMTIK_quantize_w(b_network->ih_layer.ih_wght[i][j]);
    }
    for (size_t l = 0; l < QMTIK_L; ++l){
        for (size_t i = 0; i < QMTIK_H; ++i){
            q_network->hh_layers[l].hh_bias[i] = QMTIK_quantize_w(b_network->hh_layers[l].hh_bias[i]);
            for (size_t j = 0; j < QMTIK_H; ++j) q_network->hh_layers[l].hh_wght[i][j] = QMTIK_quantize_w(b_network->hh_layers[l].hh_wght[i][j]);
        }
    }
    for (size_t i = 0; i < QMTIK_O; ++i){
        q_network->o_layer.o_bias[i] = QMTIK_quantize_w(b_network->o_layer.o_bias[i]);
        for (size_t j = 0; j < QMTIK_H; ++j) q_network->o_layer.o_wght[i][j] = QMTIK_quantize_w(b_network->o_layer.o_wght[i][j]);
    }
}
// ===============================================================
QMTIK_DEF bool QMTIK_B_store_model_to_file(QMTIK_B_Network* b_network, FILE* b_model_file) {
    if (fwrite(b_network->ih_layer.ih_bias, sizeof(QMTIK_B_Type), QMTIK_H, b_model_file) != QMTIK_H) return false;
    if (fwrite(b_network->ih_layer.ih_wght, sizeof(QMTIK_B_Type), QMTIK_H * QMTIK_I, b_model_file) != QMTIK_H * QMTIK_I) return false;
    for (size_t l = 0; l < QMTIK_L; ++l){
        if (fwrite(b_network->hh_layers[l].hh_bias, sizeof(QMTIK_B_Type), QMTIK_H, b_model_file) != QMTIK_H) return false;
        if (fwrite(b_network->hh_layers[l].hh_wght, sizeof(QMTIK_B_Type), QMTIK_H * QMTIK_H, b_model_file) != QMTIK_H * QMTIK_H) return false;
    }
    if (fwrite(b_network->o_layer.o_bias, sizeof(QMTIK_B_Type), QMTIK_O, b_model_file) != QMTIK_O) return false;
    if (fwrite(b_network->o_layer.o_wght, sizeof(QMTIK_B_Type), QMTIK_O * QMTIK_H, b_model_file) != QMTIK_O * QMTIK_H) return false;
    return true;
}
QMTIK_DEF bool QMTIK_B_load_model_from_file(FILE* b_model_file, QMTIK_B_Network* b_network) {
    if (fread(b_network->ih_layer.ih_bias, sizeof(QMTIK_B_Type), QMTIK_H, b_model_file) != QMTIK_H) return false;
    if (fread(b_network->ih_layer.ih_wght, sizeof(QMTIK_B_Type), QMTIK_H * QMTIK_I, b_model_file) != QMTIK_H * QMTIK_I) return false;
    for (size_t l = 0; l < QMTIK_L; ++l){
        if (fread(b_network->hh_layers[l].hh_bias, sizeof(QMTIK_B_Type), QMTIK_H, b_model_file) != QMTIK_H) return false;
        if (fread(b_network->hh_layers[l].hh_wght, sizeof(QMTIK_B_Type), QMTIK_H * QMTIK_H, b_model_file) != QMTIK_H * QMTIK_H) return false;
    }
    if (fread(b_network->o_layer.o_bias, sizeof(QMTIK_B_Type), QMTIK_O, b_model_file) != QMTIK_O) return false;
    if (fread(b_network->o_layer.o_wght, sizeof(QMTIK_B_Type), QMTIK_O * QMTIK_H, b_model_file) != QMTIK_O * QMTIK_H) return false;
    return true;
}
QMTIK_DEF bool QMTIK_Q_store_model_to_file(QMTIK_Q_Network* q_network, FILE* q_model_file) {
    if (fwrite(q_network->ih_layer.ih_bias, sizeof(QMTIK_Q_Type), QMTIK_H, q_model_file) != QMTIK_H) return false;
    if (fwrite(q_network->ih_layer.ih_wght, sizeof(QMTIK_Q_Type), QMTIK_H * QMTIK_I, q_model_file) != QMTIK_H * QMTIK_I) return false;
    for (size_t l = 0; l < QMTIK_L; ++l){
        if (fwrite(q_network->hh_layers[l].hh_bias, sizeof(QMTIK_Q_Type), QMTIK_H, q_model_file) != QMTIK_H) return false;
        if (fwrite(q_network->hh_layers[l].hh_wght, sizeof(QMTIK_Q_Type), QMTIK_H * QMTIK_H, q_model_file) != QMTIK_H * QMTIK_H) return false;
    }
    if (fwrite(q_network->o_layer.o_bias, sizeof(QMTIK_Q_Type), QMTIK_O, q_model_file) != QMTIK_O) return false;
    if (fwrite(q_network->o_layer.o_wght, sizeof(QMTIK_Q_Type), QMTIK_O * QMTIK_H, q_model_file) != QMTIK_O * QMTIK_H) return false;
    return true;
}
QMTIK_DEF bool QMTIK_Q_load_model_from_file(FILE* q_model_file, QMTIK_Q_Network* q_network) {
    if (fread(q_network->ih_layer.ih_bias, sizeof(QMTIK_Q_Type), QMTIK_H, q_model_file) != QMTIK_H) return false;
    if (fread(q_network->ih_layer.ih_wght, sizeof(QMTIK_Q_Type), QMTIK_H * QMTIK_I, q_model_file) != QMTIK_H * QMTIK_I) return false;
    for (size_t l = 0; l < QMTIK_L; ++l){
        if (fread(q_network->hh_layers[l].hh_bias, sizeof(QMTIK_Q_Type), QMTIK_H, q_model_file) != QMTIK_H) return false;
        if (fread(q_network->hh_layers[l].hh_wght, sizeof(QMTIK_Q_Type), QMTIK_H * QMTIK_H, q_model_file) != QMTIK_H * QMTIK_H) return false;
    }
    if (fread(q_network->o_layer.o_bias, sizeof(QMTIK_Q_Type), QMTIK_O, q_model_file) != QMTIK_O) return false;
    if (fread(q_network->o_layer.o_wght, sizeof(QMTIK_Q_Type), QMTIK_O * QMTIK_H, q_model_file) != QMTIK_O * QMTIK_H) return false;
    return true;
}
// ===============================================================
QMTIK_DEF bool QMTIK_B_load_B_sample_from_file(FILE* b_sample_file, QMTIK_B_Sample* b_sample) {
    size_t read_input = fread(b_sample->input, sizeof(QMTIK_B_Type), QMTIK_I, b_sample_file);
    size_t read_output = fread(b_sample->output, sizeof(QMTIK_B_Type), QMTIK_O, b_sample_file);
    return (read_input == QMTIK_I && read_output == QMTIK_O);
}
QMTIK_DEF bool QMTIK_B_load_Q_sample_from_file(FILE* q_sample_file, QMTIK_B_Sample* b_sample) {
    QMTIK_Q_Type i_layer_buffer[QMTIK_I];
    if (fread(i_layer_buffer, sizeof(QMTIK_Q_Type), QMTIK_I, q_sample_file) != QMTIK_I) return false;
    for (size_t i = 0; i < QMTIK_I; ++i) b_sample->input[i] = (QMTIK_B_Type)i_layer_buffer[i];
    QMTIK_Q_Type o_layer_buffer[QMTIK_O];
    if (fread(o_layer_buffer, sizeof(QMTIK_Q_Type), QMTIK_O, q_sample_file) != QMTIK_O) return false;
    for (size_t i = 0; i < QMTIK_O; ++i) b_sample->output[i] = (QMTIK_B_Type)o_layer_buffer[i];
    return true;
}
QMTIK_DEF bool QMTIK_Q_load_B_sample_from_file(FILE* b_sample_file, QMTIK_Q_Sample* q_sample) {
    QMTIK_B_Type i_layer_buffer[QMTIK_I];
    if (fread(i_layer_buffer, sizeof(QMTIK_B_Type), QMTIK_I, b_sample_file) != QMTIK_I) return false;
    for (size_t i = 0; i < QMTIK_I; ++i) q_sample->input[i] = QMTIK_quantize_a(i_layer_buffer[i]);
    QMTIK_B_Type o_layer_buffer[QMTIK_O];
    if (fread(o_layer_buffer, sizeof(QMTIK_B_Type), QMTIK_O, b_sample_file) != QMTIK_O) return false;
    for (size_t i = 0; i < QMTIK_O; ++i) q_sample->output[i] = QMTIK_quantize_a(o_layer_buffer[i]);
    return true;
}
QMTIK_DEF bool QMTIK_Q_load_Q_sample_from_file(FILE* q_sample_file, QMTIK_Q_Sample* q_sample) {
    size_t r1 = fread(q_sample->input, sizeof(QMTIK_Q_Type), QMTIK_I, q_sample_file);
    size_t r2 = fread(q_sample->output, sizeof(QMTIK_Q_Type), QMTIK_O, q_sample_file);
    return (r1 == QMTIK_I && r2 == QMTIK_O);
}
// ===============================================================
// UTIL
// ===============================================================
QMTIK_DEF QMTIK_B_Type QMTIK_B_test_from_B_samples_file(QMTIK_B_Network* b_network, FILE* test_b_samples_file) {
    QMTIK_B_Type output[QMTIK_O];
    QMTIK_B_Sample b_sample;
    size_t samples_count = 0;
    QMTIK_B_Type total_cost = 0;
    rewind(test_b_samples_file);
    while (QMTIK_B_load_B_sample_from_file(test_b_samples_file, &b_sample)) {
        QMTIK_B_load_input(b_network, b_sample.input);
        QMTIK_B_forward(b_network);
        QMTIK_B_get_output(b_network, output);
        total_cost += QMTIK_B_cost(output, b_sample.output);
        ++samples_count;
    }
    if (samples_count == 0) return 0.0f;
    return total_cost / samples_count;
}
QMTIK_DEF QMTIK_B_Type QMTIK_Q_test_from_Q_samples_file(QMTIK_Q_Network* q_network, FILE* test_q_samples_file) {
    QMTIK_Q_Type output[QMTIK_O];
    QMTIK_Q_Sample q_sample;
    size_t samples_count = 0;
    QMTIK_B_Type total_cost = 0;
    rewind(test_q_samples_file);
    while (QMTIK_Q_load_Q_sample_from_file(test_q_samples_file, &q_sample)) {
        QMTIK_Q_load_input(q_network, q_sample.input);
        QMTIK_Q_forward(q_network);
        QMTIK_Q_get_output(q_network, output);
        total_cost += QMTIK_Q_cost(output, q_sample.output);
        ++samples_count;
    }
    if (samples_count == 0) return 0.0f;
    return total_cost / samples_count;
}
// ===============================================================
#ifdef QMTIK_ENABLE_TRAINING
QMTIK_DEF void QMTIK_make_Q_model_to_file(
        QMTIK_B_Network* b_network, QMTIK_Q_Network* q_network, FILE* train_file, FILE* q_model_file, 
        size_t epochs, size_t batch_size,
        uint32_t seed, bool verbose
    ) {
    QMTIK_B_Sample b_sample;
    QMTIK_B_init_weights(b_network, seed);
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        if (verbose) printf("Epoch %zu/%zu\n", epoch + 1, epochs);
        bool load_pair_failed = false;
        rewind(train_file);
        while (true){
            for (size_t i = 0; i < batch_size; ++i) {
                if (!QMTIK_B_load_Q_sample_from_file(train_file, &b_sample)){
                    load_pair_failed = true; 
                    break;
                }
                QMTIK_B_accumulate_gradients(b_network, &b_sample);
            }
            if (b_network->adam_state.batch_count > 0) QMTIK_B_apply_gradients(b_network);
            if (load_pair_failed) break;
        }
    }
    QMTIK_B_prune(b_network);
    QMTIK_B_quantize(b_network, q_network);
    QMTIK_Q_store_model_to_file(q_network, q_model_file);
}
#endif // QMTIK_ENABLE_TRAINING

// ===============================================================

#endif // QMTIK_IMPLEMENTATION

#endif // QMTIK_H_
