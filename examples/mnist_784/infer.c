#include "qmtik_config.h"
#include "../../qmtik.h"

int main() {
    QMTIK_Q_Network q_network={0};
    FILE* q_model_file=fopen("mnist_784_model", "rb");
    if (!q_model_file){perror("Failed to open model file"); return 1;}
    if (!QMTIK_Q_load_model_from_file(q_model_file, &q_network)) {fclose(q_model_file); return 1;}
    fclose(q_model_file);

    FILE* test_file=fopen("mnist_784_test", "rb");
    if (!test_file){perror("Failed to open model file"); return 1;}
    printf("ACCURACY: %f%%\n", (1 - QMTIK_Q_test_from_Q_samples_file(&q_network, test_file)) * 100);
    fclose(test_file);

    return 0;
}
