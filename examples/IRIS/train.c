//train.c
#include "qmtik_config.h"
#define QMTIK_IMPLEMENTATION
#include "qmtik.h"

int main() {
    QMTIK_Network network={0};
    QMTIK_ReportData report_data={0};
    FILE* train_file=fopen("IRIS_train", "rb");
    if (!train_file){perror("Failed to open model file"); return 1;}
    FILE* test_file=fopen("IRIS_test", "rb");
    if (!test_file){perror("Failed to open model file"); return 1;}
    FILE* q_model_file=fopen("IRIS_model", "wb");
    if (!q_model_file){perror("Failed to open model file"); return 1;}
    QMTIK_make_model(&network, train_file, test_file, q_model_file, &report_data);
    printf("====PERFORMANCE REPORT====\n");
    printf("ACCURACY: \n");
    printf("\tBefore quantization: ~%f%%\n", report_data.bq_accuracy);
    printf("\tAfter quantization: ~%f%%\n", report_data.aq_accuracy);
    printf("MODEL SIZE: ~%zu KB (vs ~%zu KB for float32) [4x Improvement]\n", report_data.model_size/1024, (report_data.model_size*4)/1024);
    printf("INFERENCE TIME: \n");
    printf("\tBefore quantization: %f sec\n", report_data.bq_time);
    printf("\tAfter quantization: %f sec\n", report_data.aq_time);
    printf("\tImprovement: ~%.0fx\n", roundf(report_data.bq_time/report_data.aq_time));
    printf("Memory usage at training: ~%zu KB\n", report_data.train_memory/1024);
    printf("Memory usage at inference: ~%zu KB\n", report_data.infer_memory/1024);
    printf("Train time: %f sec\n", report_data.train_time);
    printf("Accuracy per epoch: \n");
    for (size_t i=0; i<QMTIK_EPOCHS; ++i) printf("%f%%,", report_data.accuracy_vs_epoch[i]);
    printf("\n");
    printf("==========================\n");
    fclose(q_model_file);
    fclose(test_file);
    fclose(train_file);
    return 0;
}
