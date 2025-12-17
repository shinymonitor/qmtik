#include "qmtik_config.h"
#define QMTIK_IMPLEMENTATION
#include "qmtik.h"

int main() {
    QMTIK_Network network={0};
    QMTIK_ReportData report_data={0};
    FILE* train_file=fopen("mnist_784_train", "rb");
    if (!train_file){perror("Failed to open model file"); return 1;}
    FILE* test_file=fopen("mnist_784_test", "rb");
    if (!test_file){perror("Failed to open model file"); return 1;}
    FILE* q_model_file=fopen("mnist_784_model", "wb");
    if (!q_model_file){perror("Failed to open model file"); return 1;}
    QMTIK_make_model(&network, train_file, test_file, q_model_file, &report_data);
    fclose(q_model_file);
    printf("====PERFORMANCE REPORT====\n");
    printf("ACCURACY: \n");
    printf("\tBefore quantization: ~%.0f%%\n", roundf(report_data.bq_accuracy));
    printf("\tAfter quantization: ~%.0f%%\n", roundf(report_data.aq_accuracy));
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

    // Accuracy per epoch plot and confusion matrix (commented out for benchmarking)
    // FILE *file = fopen("data.txt", "w");
    // if (file == NULL) {perror("Unable to open file");}
	// else{
    // 	for (int i = 0; i < QMTIK_EPOCHS; i++) {fprintf(file, "%d %f\n", i + 1, report_data.accuracy_vs_epoch[i]);}
   	// 	fclose(file);
	// }
    // FILE *gnuplot = popen("gnuplot -p", "w");
    // if (gnuplot == NULL) {perror("Error opening gnuplot");}
	// else{
	//     fprintf(gnuplot, "set title 'Accuracy vs Epoch'\n");
	//     fprintf(gnuplot, "set xlabel 'Epoch'\n");
	//     fprintf(gnuplot, "set ylabel 'Accuracy'\n");
	//     fprintf(gnuplot, "plot 'data.txt' using 1:2 with linespoints title 'Data Points'\n");
	//     pclose(gnuplot);
	// }
    //
    // int confusion_matrix[QMTIK_O][QMTIK_O];
    // QMTIK_QNetwork q_network={0};
    // q_model_file=fopen("mnist_784_model", "rb");
    // if (!q_model_file){perror("Failed to open model file"); return 1;}
    // if (!QMTIK_load_model(&q_network, q_model_file)) {fclose(q_model_file); return 1;}
    // fclose(q_model_file);
    // QMTIK_SamplePair pair;
    // rewind(test_file);
    // while (1){
    //     if (!QMTIK_load_sample_pair(test_file, &pair)) break;
    //     QMTIK_load_network_input(&q_network, pair.input);
    //     QMTIK_infer_forward(&q_network);
    //     int32_t pred_class=0;
    //     for(size_t i=1; i<QMTIK_O; ++i) if(q_network.q_o_layer.q_o_z[i]>q_network.q_o_layer.q_o_z[pred_class]) pred_class=i;
    //     int32_t exp_class=0;
    //     for(size_t i=1; i<QMTIK_O; ++i) if(pair.output[i]>pair.output[exp_class]) exp_class=i;
    //     confusion_matrix[exp_class][pred_class]++;
    // }
    // printf("Confusion Matrix:\n");
    // printf("\t");for(size_t i=0; i<QMTIK_O; ++i) printf("%d\t", (int)i); printf("\n");
    // for(size_t i=0; i<QMTIK_O; ++i){
    //     printf("%d\t", (int)i);
    //     for(size_t j=0; j<QMTIK_O; ++j) printf("%d\t", confusion_matrix[i][j]);
    //     printf("\n");
    // }

    fclose(test_file);
    fclose(train_file);
    return 0;
}
