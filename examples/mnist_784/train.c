//train.c
#include "qmtik_config.h"
#define QMTIK_IMPLEMENTATION
#include "qmtik.h"

int main() {
    QMTIK_Network network={0};
    QMTIK_init_weights(&network);
    FILE* train_file=fopen("mnist_784_train", "rb");
    if (!train_file){perror("Failed to open model file"); return 1;}
    QMTIK_train(&network, train_file);
    printf("PERFORMANCE BEFORE QUANT: %f\n", QMTIK_test_before_quant(&network, train_file));
    fclose(train_file);
    QMTIK_Model model={0};
    QMTIK_quantize_to_model(&network, &model);
    FILE* q_model_file=fopen("mnist_784_model", "wb");
    if (!q_model_file){perror("Failed to open model file"); return 1;}
    QMTIK_store_model(&model, q_model_file);
    fclose(q_model_file);
    return 0;
}
