#pragma once
#ifndef _TRAIN_H_
#define _TRAIN_H_


#include "network.h"


void back_propagation(NeuralNetwork *const pNetwork, const double *const expected);

void update_weights(NeuralNetwork *const pNetwork, const double *const X, const double learning_rate);


#endif
