#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#define MAXINPUT 100
#define EPOCH 20000
const int numInputs = 2;
const int numHiddenNodes = 2;
const int numOutputs = 1;
const int numTrainingSets = 4;

//
//  main.cpp
//  NeuralNetwork
//
//  Created by Santiago Becerra on 9/15/19.
//  Copyright © 2019 Santiago Becerra. All rights reserved.
//
//


// Simple network that can learn XOR
// Feartures : sigmoid activation function, stochastic gradient descent, and mean square error fuction

// Potential improvements :
// Different activation functions
// Batch training
// Different error funnctions
// Arbitrary number of hidden layers
// Read training end test data from a file
// Add visualization of training
// Add recurrence? (maybe that should be a separate project)

//Activation function
double sigmoid(double x) { return 1 / (1 + exp(-x)); }
double dSigmoid(double x) { return x * (1 - x); }

//Randomly produce initial weights between 0.0 and 1.0
double init_weight() { return ((double)rand()) / ((double)RAND_MAX); }

//Use bubble sort to shuffle the training sets
void shuffle(int* array, size_t n) {
    if (n > 1) {
        size_t i;
        for (i = 0; i < n - 1; i++) {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

//Process of forward propagation(including activation and sigmoid function),and result of forward output
void ForwardPropagation(int HiddenNodesNum, double* HiddenBias, int InputsNum, double* Inputs, double **HiddenWeights, double* HiddenLayer,
                        int OutputNodesNum, double* OutputBias, double** OutputWeights, double* OutputLayer)
{
    for (int j = 0; j < HiddenNodesNum; j++) {
        double activation = HiddenBias[j];
        for (int k = 0; k < InputsNum; k++) {
            activation += Inputs[k] * HiddenWeights[k][j];
        }
        HiddenLayer[j] = sigmoid(activation);
    }
    for (int j = 0; j < OutputNodesNum; j++) {
        double activation = OutputBias[j];
        for (int k = 0; k < HiddenNodesNum; k++) {
            activation += HiddenLayer[k] * OutputWeights[k][j];
        }
        OutputLayer[j] = sigmoid(activation);
    }
}
//Process of back propagation(including computing delta and update weights)
void BackPropagation(double *OutputLayer, double *HiddenLayer, double **Training_Outputs, double **Training_Inputs,
                     double *OutputBias, double *HiddenBias, double **OutputWeights, double **HiddenWeights, 
                     double learningRate,int SetNumber)
{
    // Calculate the output error after training for one time
    double *deltaOutput = (double*)malloc(sizeof(double) * numOutputs);
    for (int j = 0; j < numOutputs; j++) {
        double errorOutput = (Training_Outputs[SetNumber][j] - OutputLayer[j]);
        deltaOutput[j] = errorOutput * dSigmoid(OutputLayer[j]);
    }

    // Calculate the hidden layer error after training for one time
    double* deltaHidden = (double*)malloc(sizeof(double) * numHiddenNodes);
    for (int j = 0; j < numHiddenNodes; j++) {
        double errorHidden = 0.0f;
        for (int k = 0; k < numOutputs; k++) {
            errorHidden += deltaOutput[k] * OutputWeights[j][k];
        }
        deltaHidden[j] = errorHidden * dSigmoid(HiddenLayer[j]);
    }


    // Update the output weight through the error and bias we calculated
    for (int j = 0; j < numOutputs; j++) {
        OutputBias[j] += deltaOutput[j] * learningRate;
        for (int k = 0; k < numHiddenNodes; k++) {
            OutputWeights[k][j] += HiddenLayer[k] * deltaOutput[j] * learningRate;
        }
    }

    // Update the hidden weight through the error and bias we calculated
    for (int j = 0; j < numHiddenNodes; j++) {
        HiddenBias[j] += deltaHidden[j] * learningRate;
        for (int k = 0; k < numInputs; k++) {
            HiddenWeights[k][j] += Training_Inputs[SetNumber][k] * deltaHidden[j] * learningRate;
        }
    }
    free(deltaHidden);
    free(deltaOutput);
}

void UserInput(double* HiddenBias, double **HiddenWeights, double* HiddenLayer,
               double* OutputBias, double **OutputWeights, double* OutputLayer) 
{
    while (1) {
        int count = 0;
        //Use string to store data which user inputs,and convert them from char array to int array 
        char* test_inputStr = (char*)malloc(MAXINPUT * sizeof(char));
        double* test_data = (double*)malloc(MAXINPUT * sizeof(double));
        printf("Input:");
        scanf("%s", test_inputStr);
        while (test_inputStr[count] != '\0') {
            if (test_inputStr[count] == '0') {
                test_data[count] = '0' - '0';
            }
            else if (test_inputStr[count] == '1') {
                test_data[count] = '1' - '0';
            }
            else {
                printf("error\n");
                break;
            }
            count += 1;
        }
        free(test_inputStr);
        test_data = realloc(test_data, sizeof(double) * count);
        /*
        Set the last two bits as inputs to get the output after forward propagation,
        then replace the second to last bit with the output after first forward propagation.
        Use second to last and third to last to get the output after forward propagation,
        then replace the third to last bit with the output after second forward propagation...
        Until the inputs of forward propagation are the first bit and second bit of the array,
        the result of this forward propagation will be the answer we want.
        */
        for (count -= 1; count > 0; count--) {
            double* testing_input = (double*)malloc(numInputs * sizeof(double));
            testing_input[0] = test_data[count - 1];
            testing_input[1] = test_data[count];
            ForwardPropagation(numHiddenNodes, HiddenBias, numInputs, testing_input, HiddenWeights, HiddenLayer, numOutputs, OutputBias, OutputWeights, OutputLayer);
            test_data[count - 1] = OutputLayer[0];

            //Use round function to get 0 or 1 as final output
            if (count == 1) printf("Output:%d\n\n", (int)round(test_data[0]));
            free(testing_input);
        }
        free(test_data);
    }
}
void PrintWeights(double** HWeights,double** OWeights,double* HBias,double* OBias) {

    printf("Final Hidden Weights[ ");
    for (int j = 0; j < numHiddenNodes; j++) {
        printf("[ ");
        for (int k = 0; k < numInputs; k++) {
            printf("%lf ", HWeights[k][j]);
        }
        printf("] ");
    }
    printf("]\n");

    printf("Final Hidden Biases[ ");
    for (int j = 0; j < numHiddenNodes; j++) {
        printf("%lf ", HBias[j]);

    }
    printf("]\n");
    printf("Final Output Weights");
    for (int j = 0; j < numOutputs; j++) {
        printf("[ ");
        for (int k = 0; k < numHiddenNodes; k++) {
            printf("%lf ", OWeights[k][j]);
        }
        printf("]\n");
    }
    printf("Final Output Biases[ ");
    for (int j = 0; j < numOutputs; j++) {
        printf("%lf ", OBias[j]);

    }
    printf("]\n");
}

int main(int argc, const char* argv[]) {
    //learning rate
    const double lr = 0.1f;

    //Output of hidden layer and output layer
    double* hiddenLayer = (double*)malloc(numHiddenNodes * sizeof(double));
    double* outputLayer = (double*)malloc(numOutputs * sizeof(double));

    double* hiddenLayerBias = (double*)malloc(numHiddenNodes * sizeof(double));
    double* outputLayerBias = (double*)malloc(numOutputs * sizeof(double));

    //Use double pointer to replace the 2-dimension array 
    double** hiddenWeights = (double**)malloc(numInputs* sizeof(double*));
    for (int i = 0; i < numInputs; i++) {
        hiddenWeights[i] = (double*)malloc(numHiddenNodes * sizeof(double));
    }
    double** outputWeights = (double**)malloc(numHiddenNodes * sizeof(double*));
    for (int i = 0; i < numHiddenNodes; i++) {
        outputWeights[i] = (double*)malloc(numOutputs * sizeof(double));
    }


    //The model of XOR gate(2 inputs)
    double** training_inputs = (double**)malloc(numTrainingSets * sizeof(double*));
    for (int i = 0; i < numTrainingSets; i++) {
        training_inputs[i] = (double*)malloc(numInputs * sizeof(double));
    }
    training_inputs[0][0] = 0;
    training_inputs[0][1] = 0;
    training_inputs[1][0] = 0;
    training_inputs[1][1] = 1;
    training_inputs[2][0] = 1;
    training_inputs[2][1] = 0;
    training_inputs[3][0] = 1;
    training_inputs[3][1] = 1;

    double** training_outputs = (double**)malloc(numTrainingSets * sizeof(double*));
    for (int i = 0; i < numTrainingSets; i++) {
        training_outputs[i] = (double*)malloc(numOutputs * sizeof(double));
    }
    training_outputs[0][0] = 0;
    training_outputs[1][0] = 1;
    training_outputs[2][0] = 1;
    training_outputs[3][0] = 0;
    
    FILE* fptr;

    //initiate hiddenWeights
    for (int i = 0; i < numInputs; i++) {
        for (int j = 0; j < numHiddenNodes; j++) {
            hiddenWeights[i][j] = init_weight();
        }
    }
    //initiate hiddenLayerBias and outputWeights
    for (int i = 0; i < numHiddenNodes; i++) {
        hiddenLayerBias[i] = init_weight();
        for (int j = 0; j < numOutputs; j++) {
            outputWeights[i][j] = init_weight();
        }
    }
    //initiate outputLayerBias
    for (int i = 0; i < numOutputs; i++) {
        outputLayerBias[i] = init_weight();
    }

    //Set 4 training sets to train the model
    int* trainingSetOrder = (int*)malloc(numTrainingSets * sizeof(int));//trainingSetOrder = { 0,1,2,3 };
    for (int i = 0; i < numTrainingSets; i++) {
        trainingSetOrder[i] = i;
    }
    if ((fptr = fopen("MSE.txt", "w")) == NULL) {
        printf("Failed to open file.");
        return 0;
    }
    else {
        //Iterate through the entire training for a number of epochs=10000
        int counter = 1;
        double ErrSqu = 0;
        for (int n = 0; n < EPOCH; n++) {
            shuffle(trainingSetOrder, numTrainingSets);
            for (int x = 0; x < numTrainingSets; x++, counter++) {
                int i = trainingSetOrder[x];
                ForwardPropagation(numHiddenNodes, hiddenLayerBias, numInputs, training_inputs[i], hiddenWeights, hiddenLayer, numOutputs, outputLayerBias, outputWeights, outputLayer);
                ErrSqu += ((training_outputs[i][0] - outputLayer[0]) * (training_outputs[i][0] - outputLayer[0]));
                fprintf(fptr, "%.8lf\n", ErrSqu / counter);
                printf("Input:%lf %lf    Output:%lf    Expected Output: %lf\n", training_inputs[i][0], training_inputs[i][1], outputLayer[0], training_outputs[i][0]);
                BackPropagation(outputLayer, hiddenLayer, training_outputs, training_inputs, outputLayerBias, hiddenLayerBias, outputWeights, hiddenWeights, lr, i);
            }
        }
        printf("\n////////////////////\nloss function\n////////////////////\n");
        printf("loss= %lf\n", ErrSqu / (numTrainingSets * EPOCH));
        printf("The result of loss function is in MSE.txt\n");
    }
    fclose(fptr);
    printf("////////////////////\nTraining finished!\n////////////////////\n");

    free(training_inputs);
    free(training_outputs);

    // Print weights
    PrintWeights(hiddenWeights, outputWeights, hiddenLayerBias, outputLayerBias);
    
    //Enter Testing Data
    printf("////////////////////\nStart to test data\n////////////////////\n");
    UserInput(hiddenLayerBias, hiddenWeights, hiddenLayer,outputLayerBias, outputWeights, outputLayer);
    return 0;
}