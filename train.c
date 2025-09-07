#include "common.h"
#include <stdlib.h>
#include <time.h>
//==================================================
static inline QWghtT quantize_w(MainT x) {
    return (QWghtT)fmaxf(QWghtT_MIN, fminf(QWghtT_MAX, roundf(x/W_SCALE)));
}
static inline MainT fake_quantize_w(MainT x) {
    return quantize_w(x)*W_SCALE;
}
static inline QActvT quantize_a(MainT x) {
    return (QActvT)fmaxf(QActvT_MIN, fminf(QActvT_MAX, roundf(x/A_SCALE)));
}
static inline MainT fake_quantize_a(MainT x) {
    return quantize_a(x)*A_SCALE;
}
//==================================================
static inline void forward(Network* network) {
    for(size_t i=0; i<H; i++){
        MainT acc=network->ih_layer.ih_bias[i];
        for(size_t j=0; j<I; ++j) acc+=fake_quantize_w(network->ih_layer.ih_wght[i][j])*fake_quantize_a(network->i_layer.i_actv[j]);
        network->ih_layer.ih_z[i]=acc;
    }
    for(size_t i=0; i<H; ++i){
        MainT acc=network->hh_layers[0].hh_bias[i];
        for(size_t j=0; j<H; ++j) acc+=fake_quantize_w(network->hh_layers[0].hh_wght[i][j])*fake_quantize_a(train_activation(network->ih_layer.ih_z[j]));
        network->hh_layers[0].hh_z[i]=acc;
    }
    for(size_t l=1; l<L; ++l){
        for(size_t i=0; i<H; ++i){
            MainT acc=network->hh_layers[l].hh_bias[i];
            for(size_t j=0; j<H; j++) acc+=fake_quantize_w(network->hh_layers[l].hh_wght[i][j])*fake_quantize_a(train_activation(network->hh_layers[l-1].hh_z[j]));
            network->hh_layers[l].hh_z[i]=acc;
        }
    }
    for(size_t i=0; i<O; ++i){
        MainT acc=network->o_layer.o_bias[i];
        for(size_t j=0; j<H; ++j) acc+=fake_quantize_w(network->o_layer.o_wght[i][j])*fake_quantize_a(train_activation(network->hh_layers[L-1].hh_z[j]));
        network->o_layer.o_z[i]=acc;
    }
    train_post_process(network->o_layer.o_z);
}
//==================================================
static inline void train_step(Network* network, SamplePair sample_pair, MainT b1t, MainT b2t) {
    static MainT m_ih_w[H][I]={0}, v_ih_w[H][I]={0};
    static MainT m_ih_b[H]={0}, v_ih_b[H]={0};
    static MainT m_hh_w[L][H][H]={0}, v_hh_w[L][H][H]={0};
    static MainT m_hh_b[L][H]={0}, v_hh_b[L][H]={0};
    static MainT m_o_w[O][H]={0}, v_o_w[O][H]={0};
    static MainT m_o_b[O]={0}, v_o_b[O]={0};
    for (size_t i=0; i<I; ++i) network->i_layer.i_actv[i]=sample_pair.input[i];
    forward(network);
    static MainT dO[O];
    for (size_t i=0; i<O; i++) dO[i]=network->o_layer.o_z[i]-(MainT)sample_pair.output[i];
    static MainT dHH[L][H]={0};
    for (size_t i=0; i<H; ++i){
        MainT sum=0;
        for (size_t j=0; j<O; ++j) sum+=fake_quantize_w(network->o_layer.o_wght[j][i])*dO[j];
        dHH[L-1][i]=sum*train_activation_deriv(network->hh_layers[L-1].hh_z[i]);
    }
    for (int l=L-2; l>=0; --l){
        for (size_t i=0; i<H; ++i){
            MainT sum=0;
            for(size_t j=0; j<H; ++j) sum+=fake_quantize_w(network->hh_layers[l+1].hh_wght[j][i])*dHH[l+1][j];
            dHH[l][i]=sum*train_activation_deriv(network->hh_layers[l].hh_z[i]);
        }
    }
    static MainT dIH[H]={0};
    for (size_t i=0; i<H; ++i){
        MainT sum=0;
        for (size_t j=0; j<H; ++j) sum+=fake_quantize_w(network->hh_layers[0].hh_wght[j][i])*dHH[0][j];
        dIH[i]=sum*train_activation_deriv(network->ih_layer.ih_z[i]);
    }
    for (size_t i=0; i<H; ++i){
        MainT dB=dIH[i];
        m_ih_b[i]=BETA1*m_ih_b[i]+(1-BETA1)*dB;
        v_ih_b[i]=BETA2*v_ih_b[i]+(1-BETA2)*dB*dB;
        network->ih_layer.ih_bias[i]-=ALPHA*(m_ih_b[i]/(1-b1t))/(sqrtf(v_ih_b[i]/(1-b2t))+EPS);
        for (size_t j=0; j<I; ++j){
            MainT dW=dIH[i]*network->i_layer.i_actv[j];
            m_ih_w[i][j]=BETA1*m_ih_w[i][j]+(1-BETA1)*dW;
            v_ih_w[i][j]=BETA2*v_ih_w[i][j]+(1-BETA2)*dW*dW;
            network->ih_layer.ih_wght[i][j]-=ALPHA*(m_ih_w[i][j]/(1-b1t))/(sqrtf(v_ih_w[i][j]/(1-b2t))+EPS);
        }
    }
    for (size_t l=0; l<L; ++l){
        for (size_t i=0; i<H; ++i){
            MainT dB=dHH[l][i];
            m_hh_b[l][i]=BETA1*m_hh_b[l][i]+ (1-BETA1)*dB;
            v_hh_b[l][i]=BETA2*v_hh_b[l][i]+(1-BETA2)*dB*dB;
            network->hh_layers[l].hh_bias[i]-=ALPHA*(m_hh_b[l][i]/(1-b1t))/(sqrtf(v_hh_b[l][i]/(1-b2t))+EPS);
            for (size_t j=0; j<H; ++j){
                MainT dW=dHH[l][i]*((l==0)?train_activation(network->ih_layer.ih_z[j]):train_activation(network->hh_layers[l-1].hh_z[j]));
                m_hh_w[l][i][j]=BETA1*m_hh_w[l][i][j]+(1-BETA1)*dW;
                v_hh_w[l][i][j]=BETA2*v_hh_w[l][i][j]+(1-BETA2)*dW*dW;
                network->hh_layers[l].hh_wght[i][j]-=ALPHA*(m_hh_w[l][i][j]/(1-b1t))/(sqrtf(v_hh_w[l][i][j]/(1-b2t))+EPS);
            }
        }
    }
    for (size_t i=0; i<O; ++i){
        MainT dB=dO[i];
        m_o_b[i]=BETA1*m_o_b[i]+(1-BETA1)*dB;
        v_o_b[i]=BETA2*v_o_b[i]+(1-BETA2)*dB*dB;
        network->o_layer.o_bias[i]-=ALPHA*(m_o_b[i]/(1-b1t))/(sqrtf(v_o_b[i]/(1-b2t))+EPS);
        for (size_t j=0; j<H; ++j){
            MainT dW=dO[i]*train_activation(network->hh_layers[L-1].hh_z[j]);
            m_o_w[i][j]=BETA1*m_o_w[i][j]+(1-BETA1)*dW;
            v_o_w[i][j]=BETA2*v_o_w[i][j]+(1-BETA2)*dW*dW;
            network->o_layer.o_wght[i][j]-=ALPHA*(m_o_w[i][j]/(1-b1t))/(sqrtf(v_o_w[i][j]/(1-b2t))+EPS);
        }
    }
}
//==================================================
void quantize_to_model(Network* network, Model* model){
    for (size_t i=0; i<H; ++i){
        model->q_ih_bias[i]=quantize_w(network->ih_layer.ih_bias[i]);
        for (size_t j=0; j<I; ++j) model->q_ih_wght[i][j]=quantize_w(network->ih_layer.ih_wght[i][j]);
    }
    for (size_t l=0; l<L; ++l){
        for (size_t i=0; i<H; ++i){
            model->q_hh_biases[l][i]=quantize_w(network->hh_layers[l].hh_bias[i]);
            for (size_t j=0; j<H; ++j) model->q_hh_wghts[l][i][j]=quantize_w(network->hh_layers[l].hh_wght[i][j]);
        }
    }
    for (size_t i=0; i<O; ++i){
        model->q_o_bias[i]=quantize_w(network->o_layer.o_bias[i]);
        for (size_t j=0; j<H; ++j) model->q_o_wght[i][j]=quantize_w(network->o_layer.o_wght[i][j]);
    }
}
void store_model(Model* model, const char* filename){
    FILE* f=fopen(filename, "wb");
    if (!f){perror("Failed to open model file"); return;}
    fwrite(model, sizeof(Model), 1, f);
    fclose(f);
}
//==================================================
int main() {
    Network network={0};
    srand(time(NULL));
    for (size_t i=0; i<H; ++i){
        network.ih_layer.ih_bias[i]=0.0f;
        for (size_t j=0; j<I; ++j) network.ih_layer.ih_wght[i][j]=((MainT)rand()/RAND_MAX-0.5f)*2.0f*sqrtf(1.0f/I);
    }
    for (size_t l=0; l<L; ++l){
        for (size_t i=0; i<H; ++i) {
            network.hh_layers[l].hh_bias[i]=0.0f;
            for (size_t j=0; j<H; ++j) network.hh_layers[l].hh_wght[i][j]=((MainT)rand()/RAND_MAX-0.5f)*2.0f*sqrtf(1.0f/H);
        }
    }
    for (size_t i=0; i<O; ++i){
        network.o_layer.o_bias[i]=0.0f;
        for (size_t j=0; j<H; ++j) network.o_layer.o_wght[i][j]=((MainT)rand()/RAND_MAX-0.5f)*2.0f*sqrtf(1.0f/H);
    }
    //==================================================
    SamplePair batch[BATCH_SIZE];
    //==================================================
    FILE* train_file=fopen(TRAIN_FILE, "rb");
    if (!train_file){perror("Failed to open model file"); return 0;}
    size_t t=0;
    MainT b1t=1, b2t=1;
    uint8_t load_pair_failed=0;
    int _sample_number=0;
    for (int _epoch=0; _epoch<EPOCHS; ++_epoch){
        if (_epoch%EPOCHS_DEBUG_UPDATE_POINT==0) printf("EPOCH: %d\n", _epoch);
        rewind(train_file);
        _sample_number=0;
        load_pair_failed=0;
        while (1){
            if (_sample_number%(SAMPLE_NUMBER_DEBUG_UPDATE_POINT*BATCH_SIZE)==0) printf("SAMPLE_NUMBER: %d\n", _sample_number);
            for (size_t i=0; i<BATCH_SIZE; ++i) if (!load_sample_pair(train_file, &batch[i])){load_pair_failed = 1; break;}
            if (load_pair_failed) break;
            _sample_number+=BATCH_SIZE;
            for (size_t i=0; i<BATCH_SIZE; ++i) {
                t++;
                b1t *= BETA1;
                b2t *= BETA2;
                train_step(&network, batch[i], b1t, b2t);
            }
        }
    }
    //==================================================
    SamplePair pair;
    uint64_t total_cost=0;
    _sample_number=0;
    rewind(train_file);
    while (1){
        if (!load_sample_pair(train_file, &pair)) break;
        for(size_t i=0; i<I; ++i) network.i_layer.i_actv[i]=(MainT)pair.input[i];
        forward(&network);
        int32_t temp_cost=train_cost(&network, pair.output);
        #ifdef TRAIN_DEBUG
            if (_sample_number%(SAMPLE_NUMBER_DEBUG_UPDATE_POINT*BATCH_SIZE)==0) {
                printf("SAMPLE_NUMBER: %d\n", _sample_number);
                for(size_t i=0; i<O; ++i) printf("%f,", network.o_layer.o_z[i]);
                printf("\n");
                for(size_t i=0; i<O; ++i) printf("%d,", pair.output[i]);
                printf("\n");
                printf("%d\n", temp_cost);
            }
        #endif
        _sample_number+=1;
        total_cost+=temp_cost;
    }
    printf("PERFORMANCE BEFORE QUANT: %.2f\n", (MainT)total_cost/_sample_number);
    fclose(train_file);
    //==================================================
    Model model={0};
    quantize_to_model(&network, &model);
    store_model(&model, MODEL_FILE);
    return 0;
}
