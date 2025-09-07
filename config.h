#define I 2
#define H 2
#define L 1
#define O 1

#define TRAIN_FILE "<training_samples_file_here>"
#define INFER_FILE "<infering_samples_file_here>"
#define MODEL_FILE "<model_file_here>"

#define RELU_ACTV
//#define LEAKY_RELU_ACTV
//#define SIGMOID_ACTV
//#define TANH_ACTV
#define LINEAR_PP
//#define SOFT_MAX_PP
//#define SIGMOID_PP
#define MSE_COST
//#define CROSS_ENTROPY_COST

#define BATCH_SIZE 1
#define ALPHA 0.001f
#define EPOCHS 8

#define W_SCALE 0.05f
#define A_SCALE 0.5f
#define BETA1 0.9f
#define BETA2 0.999f
#define EPS 1e-8f

#define EPOCHS_DEBUG_UPDATE_POINT 1
#define SAMPLE_NUMBER_DEBUG_UPDATE_POINT 1024
#define TRAIN_DEBUG
#define INFER_DEBUG