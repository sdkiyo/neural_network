#pragma once
#ifndef _NETWORK_H_
#define _NETWORK_H_


#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#include "colors.h"


#define FILL_MIN -0.10
#define FILL_MAX 0.10


typedef void (*PFN_activation_callback)(double *const);
typedef double (*PFN_derivative_callback)(const double);
typedef double (*PFN_loss_callback)(const double, const double);


typedef struct NeuralNetwork {
	double ***weights;
	double **biases;
	double **outputs;
	double **errors;
	uint32_t *inputs_size;
	uint32_t *outputs_size;
	PFN_activation_callback activation_callback;
	PFN_derivative_callback derivative_callback;
	PFN_loss_callback loss_callback;
	uint32_t layers_count;
} NeuralNetwork;


typedef void (*PFN_forward_network)(NeuralNetwork *const pNetwork, const double *const pX);
typedef void (*PFN_init_network)(NeuralNetwork *const pNetwork, const uint32_t X_size, const uint32_t layers_count, const uint32_t *const pLayers_output_size, PFN_activation_callback activation_callback, PFN_derivative_callback derivative_callback, PFN_loss_callback loss_callback);
typedef void (*PFN_free_network)(NeuralNetwork *const pNetwork);
typedef void (*PFN_clear_network)(NeuralNetwork *const pNetwork);



#endif
