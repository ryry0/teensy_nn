#define NUM_LAYERS 3
#define INPUT_LAYER_SIZE  784 //have to make sure you can map data to inputs
#define OUTPUT_LAYER_SIZE 10

#define PIC_WIDTH 28
#define PIC_HEIGHT 28
#define PICTURE_SIZE 784 //num pixels 28*28

#define NUM_PICTURES 60000
#define NUM_SAMPLES 50000
#define VERIF_SAMPLES 10000

#define TRAIN_OFFSET      0x10
#define TRAIN_EXP_OFFSET  0x08

#define DEFAULT_TRAIN     "data/train-images-idx3-ubyte"
#define DEFAULT_EXPECT    "data/train-labels-idx1-ubyte"

#include "nn.h"

void setup() {
  neural_network_t neural_net;
  size_t layer_sizes[NUM_LAYERS] = {INPUT_LAYER_SIZE,30,OUTPUT_LAYER_SIZE};
  initNNet(&neural_net, NUM_LAYERS, layer_sizes);

  destroyNNet(&neural_net);
}

void loop() {
}
