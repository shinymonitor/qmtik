//infer.c
#include "qmtik_config.h"
#define QMTIK_IMPLEMENTATION
#include "qmtik.h"

int main() {
    QMTIK_QNetwork q_network={0};
    FILE* q_model_file=fopen("mnist_784_model", "rb");
    if (!q_model_file){perror("Failed to open model file"); return 1;}
    if (!QMTIK_load_model(&q_network, q_model_file)) {fclose(q_model_file); return 1;}
    fclose(q_model_file);
    FILE* test_file=fopen("mnist_784_test", "rb");
    if (!test_file){perror("Failed to open model file"); return 1;}
    printf("PERFORMANCE: %f%%\n", QMTIK_test_after_quant(&q_network, test_file));
    fclose(test_file);
    return 0;
}
