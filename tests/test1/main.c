#include <stdio.h>

#include "../../include/train.h"
#include "../../include/network.h"
#include "../../include/print_network.h"


int main(const int argc, const char *const *const argv)
{
	printf(TRANSPARENT_COLOR "start." RESET_COLOR "\n");


	const double X[] = {1.0, 2.0, 3.0, 2.5};
	const uint32_t x_size = 4;

	const uint32_t layers_output[] = {3, 4, 2};
	const uint32_t layers_count = 3;
	NeuralNetwork nn = {};

	init_network(&nn, x_size, layers_count, layers_output, activation_sigmoid);

	forward_network(&nn, X, true, true);

	print_last_layer(&nn, true, true, true, true, true);
	//print_network(&nn, true, true, true, true, true, true, true);

	free_network(&nn);


	printf(TRANSPARENT_COLOR "end." RESET_COLOR "\n");
	return 0;
}
