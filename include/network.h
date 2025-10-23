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


typedef void (*PFN_activation_callback)(double *const x);

typedef struct NeuralNetwork {
	double ***weights;
	double **biases;
	double **outputs;
	double **errors;
	uint32_t *inputs_size;
	uint32_t *outputs_size;
	PFN_activation_callback activation_callback;
	uint32_t layers_count;
} NeuralNetwork;


void activation_sigmoid(double *const x);

void activation_ReLU(double *const x);

void forward_network(NeuralNetwork *const pNetwork, const double *const pX);

void init_network(NeuralNetwork *const pNetwork, const uint32_t X_size, const uint32_t layers_count, const uint32_t *const pLayers_output_size, PFN_activation_callback activation_callback);

void free_network(NeuralNetwork *const pNetwork);

void clear_network(NeuralNetwork *const pNetwork);


#endif
