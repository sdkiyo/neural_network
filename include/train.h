#pragma once
#ifndef _TRAIN_H_
#define _TRAIN_H_

#include "network.h"


void back_propagation(NeuralNetwork *const pNetwork, const double *const pX, const double *const expected);


#endif
