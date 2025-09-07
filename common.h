#include "config.h"
//==================================================
#include <stdio.h>
#include <stdint.h>
#include <math.h>
//==================================================
#define MainT float
#define QWghtT int8_t
#define QActvT int8_t
#define QWghtT_MAX 127
#define QWghtT_MIN -128
#define QActvT_MAX 127
#define QActvT_MIN -128
//==================================================
typedef struct {MainT i_actv[I];} ILayer;
typedef struct {MainT ih_z[H]; MainT ih_wght[H][I], ih_bias[H];} IHLayer;
typedef struct {MainT hh_z[H]; MainT hh_wght[H][H], hh_bias[H];} HHLayer;
typedef struct {MainT o_z[O]; MainT o_wght[O][H], o_bias[O];} OLayer;
typedef struct {ILayer i_layer; IHLayer ih_layer; HHLayer hh_layers[L]; OLayer o_layer;} Network;
typedef struct {QActvT input[I], output[O];} SamplePair;
typedef struct {QWghtT q_ih_wght[H][I], q_ih_bias[H], q_hh_wghts[L][H][H], q_hh_biases[L][H], q_o_wght[O][H], q_o_bias[O];} Model;
typedef struct {QActvT q_i_actv[I];} QILayer;
typedef struct {QActvT q_ih_actv[H]; QWghtT q_ih_wght[H][I], q_ih_bias[H];} QIHLayer;
typedef struct {QActvT q_hh_actv[H]; QWghtT q_hh_wght[H][H], q_hh_bias[H];} QHHLayer;
typedef struct {QActvT q_o_z[O]; QWghtT q_o_wght[O][H], q_o_bias[O];} QOLayer;
typedef struct {QILayer q_i_layer; QIHLayer q_ih_layer; QHHLayer q_hh_layers[L]; QOLayer q_o_layer;} QNetwork;
//==================================================
static inline int load_sample_pair(FILE* file, SamplePair* pair) {
    size_t r1=fread(pair->input, 1, I, file);
    size_t r2=fread(pair->output, 1, O, file);
    return (r1==I&&r2==O);
}
//==================================================
#ifdef RELU_ACTV
    static inline MainT train_activation(MainT x) {return x>0?x:0.0f;}
    static inline MainT train_activation_deriv(MainT x) {return x>0?1.0f:0.0f;}
    static inline QActvT infer_activation(MainT x) {return (QActvT)fmaxf(QActvT_MIN, fminf(QActvT_MAX, roundf(train_activation(x)/A_SCALE)));}
#endif
#ifdef LEAKY_RELU_ACTV
    #define LEAK 0.01f
    static inline MainT train_activation(MainT x) {return x>0?x:LEAK*x;}
    static inline MainT train_activation_deriv(MainT x) {return x>0?1.0f:LEAK;}
    static inline QActvT infer_activation(float x) {return (QActvT)fmaxf(QActvT_MIN, fminf(QActvT_MAX, roundf(train_activation(x)/A_SCALE)));}
#endif
#ifdef SIGMOID_ACTV
    #define CLAMP_MIN -88.0f
    #define CLAMP_MAX 88.0f
    static inline MainT train_activation(MainT x) {return 1.0f/(1.0f+expf(-(fmaxf(CLAMP_MIN, fminf(CLAMP_MAX, x)))));}    
    static inline MainT train_activation_deriv(MainT x) {return train_activation(x)*(1.0f-train_activation(x));}
    static inline QActvT infer_activation(MainT x) {return (QActvT)fmaxf(QActvT_MIN, fminf(QActvT_MAX, roundf(train_activation(x)/A_SCALE)));}
#endif
#ifdef TANH_ACTV
    #define CLAMP_MIN -88.0f
    #define CLAMP_MAX 88.0f
    static inline MainT train_activation(MainT x) {return tanhf(fmaxf(CLAMP_MIN, fminf(CLAMP_MAX, x)));}
    static inline MainT train_activation_deriv(MainT x) {return 1.0f - train_activation(x) * train_activation(x);}
    static inline QActvT infer_activation(MainT x) {return (QActvT)fmaxf(QActvT_MIN, fminf(QActvT_MAX, roundf(train_activation(x)/A_SCALE)));}
#endif
//==================================================
#ifdef LINEAR_PP
    static inline void train_post_process(MainT z[O]) {for (size_t i=0; i<O; ++i) z[i]=fmaxf(-127.0f, fminf(127.0f, z[i]));}
    static inline void infer_post_process(QActvT z[O]) {(void)z;}
#endif
#ifdef SOFT_MAX_PP
    static inline void train_post_process(MainT z[O]) {
        MainT max_z=z[0];
        for (size_t i=1; i<O; ++i) if (z[i]>max_z) max_z=z[i];
        MainT sum=0.0f;
        for (size_t i=0; i<O; ++i) {z[i]=expf(z[i]-max_z); sum+=z[i];}
        for (size_t i=0; i<O; ++i) z[i]=(z[i]/sum)*127;
    }
    static inline void infer_post_process(QActvT z[O]) {
        float temp[O];
        float max_z=z[0]*A_SCALE;
        for (size_t i=1; i<O; ++i) if (z[i]*A_SCALE>max_z) max_z=z[i]*A_SCALE;
        float sum = 0.0f;
        for (size_t i=0; i<O; ++i) {temp[i]=expf(z[i]*A_SCALE-max_z); sum+=temp[i];}
        for (size_t i=0; i<O; ++i) {z[i]=(QActvT)roundf((temp[i]/sum)*127);}
    }
#endif
#ifdef SIGMOID_PP
    #define CLAMP_MIN -88.0f
    #define CLAMP_MAX 88.0f
    static inline void train_post_process(MainT z[O]) {for (size_t i=0; i<O; ++i){z[i]=fmaxf(CLAMP_MIN, fminf(CLAMP_MAX, z[i])); z[i]=(1.0f/(1.0f+expf(-z[i])))*127.0f;}}
    static inline void infer_post_process(QActvT z[O]) {for (size_t i=0; i<O; ++i) z[i]=(QActvT)roundf((1.0f/(1.0f+expf(-fmaxf(CLAMP_MIN, fminf(CLAMP_MAX, z[i]*A_SCALE)))))*127.0f);}
#endif
//==================================================
#ifdef MSE_COST
    static inline MainT train_cost(Network* network, QActvT expected[O]) {
        MainT total_error=0.0f;
        for (size_t i=0; i<O; ++i) {MainT diff=network->o_layer.o_z[i]-(MainT)expected[i]; total_error+=diff*diff;}
        return total_error/O;
    }
    static inline MainT infer_cost(QNetwork* q_network, QActvT expected[O]) {
        MainT total_error=0.0f;
        for (size_t i = 0; i < O; ++i) {MainT diff=(MainT)q_network->q_o_layer.q_o_z[i]-(MainT)expected[i]; total_error+=diff*diff;}
        return total_error/O;
    }
#endif
#ifdef CROSS_ENTROPY_COST
    static inline int8_t train_cost(Network* network, QActvT expected[O]){
        int32_t pred_class=0;
        for(size_t i=1; i<O; ++i) if (network->o_layer.o_z[i]>network->o_layer.o_z[pred_class]) pred_class=i;
        int32_t exp_class=0;
        for(size_t i=1; i<O; ++i) if(expected[i]>expected[exp_class]) exp_class=i;
        return (pred_class==exp_class)?1:0;
    }
    static inline int8_t infer_cost(QNetwork* q_network, QActvT expected[O]){
        int32_t pred_class=0;
        for(size_t i=1; i<O; ++i) if(q_network->q_o_layer.q_o_z[i]>q_network->q_o_layer.q_o_z[pred_class]) pred_class=i;
        int32_t exp_class=0;
        for(size_t i=1; i<O; ++i) if(expected[i]>expected[exp_class]) exp_class=i;
        return (pred_class==exp_class)?1:0;
    }
#endif
