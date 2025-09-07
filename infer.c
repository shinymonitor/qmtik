#include "common.h"
//==================================================
void load_model(QNetwork* q_network, const char* filename) {
    FILE* f=fopen(filename, "rb");
    if (!f){perror("Failed to open model file"); return;}
    Model model;
    if (!fread(&model, sizeof(Model), 1, f)){perror("Failed to read model file"); return;}
    fclose(f);
    for (size_t i=0; i<H; ++i){
        q_network->q_ih_layer.q_ih_bias[i]=model.q_ih_bias[i];
        for (size_t j=0; j<I; ++j) q_network->q_ih_layer.q_ih_wght[i][j]=model.q_ih_wght[i][j];
    }
    for (size_t l=0; l<L; ++l){
        for (size_t i=0; i<H; ++i){
            q_network->q_hh_layers[l].q_hh_bias[i]=model.q_hh_biases[l][i];
            for (size_t j=0; j<H; ++j) q_network->q_hh_layers[l].q_hh_wght[i][j]=model.q_hh_wghts[l][i][j];
        }
    }
    for(size_t i=0; i<O; ++i){
        q_network->q_o_layer.q_o_bias[i]=model.q_o_bias[i];
        for(size_t j=0; j<H; ++j) q_network->q_o_layer.q_o_wght[i][j]=model.q_o_wght[i][j];
    }
}
//==================================================
static inline void forward(QNetwork* q_network) {
    MainT acc;
    for (size_t i=0; i<H; ++i){
        acc=q_network->q_ih_layer.q_ih_bias[i]*W_SCALE;
        for (size_t j=0; j<I; ++j) acc+=(q_network->q_ih_layer.q_ih_wght[i][j]*W_SCALE)*(q_network->q_i_layer.q_i_actv[j]*A_SCALE);
        q_network->q_ih_layer.q_ih_actv[i]=infer_activation(acc);
    }
    for (size_t i=0; i<H; ++i){
        acc=q_network->q_hh_layers[0].q_hh_bias[i]*W_SCALE;
        for (size_t j=0; j<H; ++j) acc+=(q_network->q_hh_layers[0].q_hh_wght[i][j]*W_SCALE)*(q_network->q_ih_layer.q_ih_actv[j]*A_SCALE);
        q_network->q_hh_layers[0].q_hh_actv[i]=infer_activation(acc);
    }
    for (size_t l=1; l<L; ++l){
        for (size_t i=0; i<H; ++i){
            acc=q_network->q_hh_layers[l].q_hh_bias[i]*W_SCALE;
            for (size_t j=0; j<H; ++j) acc+=(q_network->q_hh_layers[l].q_hh_wght[i][j]*W_SCALE)*(q_network->q_hh_layers[l-1].q_hh_actv[j]*A_SCALE);
            q_network->q_hh_layers[l].q_hh_actv[i]=infer_activation(acc);
        }
    }
    for (size_t i=0; i<O; ++i){
        acc=q_network->q_o_layer.q_o_bias[i]*W_SCALE;
        for (size_t j=0; j<H; ++j) acc+=(q_network->q_o_layer.q_o_wght[i][j]*W_SCALE)*(q_network->q_hh_layers[L-1].q_hh_actv[j]*A_SCALE);
        q_network->q_o_layer.q_o_z[i]=(QActvT)fmaxf(QActvT_MIN, fminf(QActvT_MAX, roundf(acc/A_SCALE)));
    }
    infer_post_process(q_network->q_o_layer.q_o_z);
}
//==================================================
int main() {
    QNetwork net={0};
    load_model(&net, MODEL_FILE);
    FILE* infer_file=fopen(INFER_FILE, "rb");
    if (!infer_file){perror("Failed to open model file"); return 0;}
    SamplePair pair;
    uint64_t total_cost=0;
    int _sample_number=0;
    while (1){
        if (!load_sample_pair(infer_file, &pair)) break;
        for (size_t i=0; i<I; ++i) net.q_i_layer.q_i_actv[i]=pair.input[i];
        forward(&net);
        int32_t temp_cost=infer_cost(&net, pair.output);
        #ifdef INFER_DEBUG
            if (_sample_number%(SAMPLE_NUMBER_DEBUG_UPDATE_POINT*BATCH_SIZE)==0){
                printf("SAMPLE_NUMBER: %d\n", _sample_number);
                for (size_t i=0; i<O; ++i) printf("%d,",net.q_o_layer.q_o_z[i]);
                printf("\n");
                for (size_t i=0; i<O; ++i) printf("%d,",pair.output[i]);
                printf("\n");
                printf("%d\n", temp_cost);
            }
        #endif
        _sample_number+=1;
        total_cost+=temp_cost;
    }
    printf("PERFORMANCE: %.2f\n", (MainT)total_cost/_sample_number);
    fclose(infer_file);
    return 0;
}
