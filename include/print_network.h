#pragma once
#ifndef _PRINT_NETWORK_H_
#define _PRINT_NETWORK_H_


#include "network.h"


typedef void (*PFN_print_layer)(NeuralNetwork *const pNetwork, const uint32_t layer_number, const bool showNeuronErrors, const bool showInput, const bool showWeights, const bool showBiases, const bool showOuts);
typedef void (*PFN_print_last_layer)(NeuralNetwork *const pNetwork, const bool showNeuronErrors, const bool showInput, const bool showWeights, const bool showBiases, const bool showOuts);
typedef void (*PFN_print_network)(NeuralNetwork *const pNetwork, const bool showLayersCount, const bool showActivationFunction, const bool showNeuronErrors, const bool showInput, const bool showWeights, const bool showBiases, const bool showOuts);


#endif
