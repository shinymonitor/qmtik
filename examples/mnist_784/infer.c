#include "qmtik_config.h"
#define QMTIK_IMPLEMENTATION
#include "qmtik.h"

int main() {
    QMTIK_QNetwork q_network={0};
    FILE* q_model_file=fopen("mnist_784_model", "rb");
    if (!q_model_file){perror("Failed to open model file"); return 1;}
    if (!QMTIK_load_model(&q_network, q_model_file)) {fclose(q_model_file); return 1;}
    fclose(q_model_file);
    printf("Memory usage at inference: ~%zu KB\n", QMTIK_get_inference_memory_usage()/1024);

    FILE* test_file=fopen("mnist_784_test", "rb");
    if (!test_file){perror("Failed to open model file"); return 1;}
    printf("PERFORMANCE: %f%%\n", QMTIK_test_after_quant(&q_network, test_file));
    fclose(test_file);

    // Demo inference (commented out for benchmarking)
    // FILE* demo_file=fopen("demo_sample", "rb");
    // if (!demo_file){perror("Failed to open model file"); return 1;}
    // QMTIK_SamplePair demo_pair={0};
    // QMTIK_load_sample_pair(demo_file, &demo_pair);
    // QMTIK_load_network_input(&q_network, demo_pair.input);
    // QMTIK_infer_forward(&q_network);
    // QMTIK_QActvT output[QMTIK_O]={0};
    // QMTIK_get_network_output(&q_network, output);
    // printf("DEMO OUTPUT: \n");
    // for(size_t i=0; i<QMTIK_O; ++i) {
    //     size_t probability = output[i]*100/127;
    //     printf("%zu: ", i);
    //     for (size_t j=0; j<probability; ++j) printf("|");
    //     printf(" %zu %% (%d)\n", probability, output[i]);
    // }
    // printf("\n");
    // fclose(demo_file);

    return 0;
}
