#define I 784
#define H 256
#define L 2
#define O 10

#define TRAIN_FILE "mnist_784_train"
#define INFER_FILE "mnist_784_infer"
#define MODEL_FILE "mnist_784_model"

#define LEAKY_RELU_ACTV
#define SOFT_MAX_PP
#define CROSS_ENTROPY_COST

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