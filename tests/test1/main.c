#include <stdio.h>

#include "../../include/train.h"
#include "../../include/network.h"
#include "../../include/print_network.h"


int main(const int argc, const char *const *const argv)
{
	printf(TRANSPARENT_COLOR "start." RESET_COLOR "\n");


	const double X[3][4] = {
		{1.0, 1.0, 0.0, 0.0},
		{1.0, 1.0, 1.0, 1.0},
		{0.0, 0.0, 1.0, 1.0}
	};
	const uint32_t x_size = 4;
	const uint32_t data_size = 3;

	const uint32_t layers_output[] = {5, 10, 4, 7, 4};
	const uint32_t layers_count = 5;
	NeuralNetwork nn = {};

	init_network(&nn, x_size, layers_count, layers_output, activation_sigmoid);

	//forward_network(&nn, X[0], true, true);

	//print_last_layer(&nn, true, true, true, true, true);
	//print_network(&nn, true, true, true, true, true, true, true);

	const double expected[3][4] = {
		{0.0, 0.0, 1.0, 1.0},
		{1.0, 1.0, 1.0, 1.0},
		{1.0, 1.0, 0.0, 0.0}
	};

	for (uint32_t i = 0; i < 300'000; i++)
	{
		for (uint32_t j = 0; j < data_size; j++)
		{
			forward_network(&nn, X[j], true, true);
			back_propagation(&nn, expected[j]);
			update_weights(&nn, X[j], 0.0001);
		}
	}
	printf("\n");
	forward_network(&nn, X[0], true, true);
	print_last_layer(&nn, true, true, true, true, true);
	printf("\n");
	forward_network(&nn, X[1], true, true);
	print_last_layer(&nn, true, true, true, true, true);
	printf("\n");
	forward_network(&nn, X[2], true, true);
	print_last_layer(&nn, true, true, true, true, true);

	free_network(&nn);


	printf(TRANSPARENT_COLOR "end." RESET_COLOR "\n");
	return 0;
}
