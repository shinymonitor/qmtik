#include "qmtik_config.h"
#define QMTIK_ENABLE_TRAINING
#include "../../qmtik.h"

#include <time.h>

int main() {
    QMTIK_B_Network b_network = {0};
    QMTIK_Q_Network q_network = {0};
    FILE* train_file=fopen("fashion-mnist_train", "rb");
    if (!train_file){perror("Failed to open model file"); return 1;}
    FILE* q_model_file=fopen("fashion-mnist_model", "wb");
    if (!q_model_file){perror("Failed to open model file"); return 1;}
    QMTIK_make_Q_model_to_file(&b_network, &q_network, train_file, q_model_file, 8, 32, time(NULL), true);
    fclose(q_model_file);
    fclose(train_file);
    return 0;
}
