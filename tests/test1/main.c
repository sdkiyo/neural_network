#include <stdio.h>
#include <time.h>

#include "../../include/train.h"
#include "../../include/network.h"
#include "../../include/print_network.h"


int main(const int argc, const char *const *const argv)
{
	printf(TRANSPARENT_COLOR "start." RESET_COLOR "\n");
	srand(time(NULL));


	const double X[4][2] = {
		{0.0, 0.0},
		{1.0, 0.0},
		{1.0, 1.0},
		{0.0, 1.0}
	};
	const uint32_t x_size = 2;
	const uint32_t data_size = 4;

	const uint32_t layers_output[] = {2, 2, 2};
	const uint32_t layers_count = 3;
	NeuralNetwork nn = {};

	init_network(&nn, x_size, layers_count, layers_output, activation_sigmoid);

	//print_network(&nn, true, true, true, true, true, true, true);

	const double expected[4][2] = {
		{0.0, 0.0},
		{0.0, 1.0},
		{1.0, 1.0},
		{1.0, 0.0}
	};

	uint32_t g = 0;
	for (uint32_t i = 0; i < 1; i++)
	{
		for (uint32_t k = 0; k < 1'000; k++)
		{
			for (uint32_t j = 0; j < data_size; j++)
			{
				g = rand() % data_size;
				forward_network(&nn, X[g]);
				back_propagation(&nn, expected[g]);
				update_weights(&nn, X[g], 0.4);
			}
		}
	}

	printf(YELLOW "after train:\n" RESET_COLOR);
	for (uint32_t i = 0; i < data_size; i++)
	{
		forward_network(&nn, X[i]);
		if (i == 0)
		{
			printf("\n");
			print_network(&nn, false, false, true, false, true, true, true);
			printf("\n");
		}
		print_last_layer(&nn, false, false, false, false, true);
	}

	free_network(&nn);


	printf(TRANSPARENT_COLOR "end." RESET_COLOR "\n");
	return 0;
}
