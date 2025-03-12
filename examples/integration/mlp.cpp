#include "layers/linear.h"
#include "layers/relu.h"
#include "layers/add.h"
#include "layers/softmax.h"



int main() { 
    int batch_size = 1; 
    int in_features = 128; 
    int hidden_dim = 256; 
    int out_features = 10; 

    float *buffer_A = new float[batch_size * std::max(std::max(hidden_dim, out_features), in_features)];
    float *buffer_B = new float[batch_size * std::max(std::max(hidden_dim, out_features), in_features)];

    float *W1_data = new float[in_features * hidden_dim]; 
    float *W2_data = new float[hidden_dim * hidden_dim]; 
    float *W3_data = new float[hidden_dim * out_features]; 

    T_FP* in_tensor = new T_FP(buffer_A, batch_size, in_features); 

    T_FP* out1_tensor = new T_FP(buffer_B, batch_size, hidden_dim); 
    T_FP* out2_tensor = new T_FP(buffer_A, batch_size, hidden_dim); 
    T_FP* out3_tensor = new T_FP(buffer_B, batch_size, out_features); 

    T_FP* W1_tensor = new T_FP(W1_data, hidden_dim, in_features); 
    T_FP* W2_tensor = new T_FP(W2_data, hidden_dim, hidden_dim); 
    T_FP* W3_tensor = new T_FP(W3_data, out_features, hidden_dim); 

    Linear* layer1 = new Linear(in_features, hidden_dim); 
    Linear* layer2 = new Linear(hidden_dim, hidden_dim); 
    Linear* layer3 = new Linear(hidden_dim, out_features); 

    layer1->set_weight(W1_tensor); 
    layer2->set_weight(W2_tensor); 
    layer3->set_weight(W3_tensor); 

    layer1->forward(out1_tensor, in_tensor); 
    layer2->forward(out2_tensor, out1_tensor); 
    layer3->forward(out3_tensor, out2_tensor); 

    delete[] buffer_A;
    delete[] buffer_B;
    delete in_tensor;
    delete out1_tensor;
    delete out2_tensor;
    delete out3_tensor;
    delete W1_tensor;
    delete W2_tensor;
    delete W3_tensor;
    delete layer1;
    delete layer2;
    delete layer3;
}