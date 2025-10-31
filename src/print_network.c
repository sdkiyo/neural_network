#include "print_network.h"


void print_layer(NeuralNetwork *const pNetwork, const uint32_t layer_number, const bool showNeuronErrors, const bool showInput, const bool showWeights, const bool showBiases, const bool showOuts)
{
	printf("layer %d\n" RESET_COLOR, layer_number);

	if (showInput == true)
	{
		printf(RESET_COLOR "input: ");

		if (layer_number != 0)
		{
			for (uint32_t j = 0; j < pNetwork->inputs_size[layer_number]; j++)
			{
				printf("%f", pNetwork->outputs[layer_number-1][j]);
			}
		}
		else
		{
			printf("it's first layer.");
		}

		printf("\n" RESET_COLOR);
	}

	for (uint32_t j = 0; j < pNetwork->outputs_size[layer_number]; j++)
	{
		printf(BLUE "n[%d]: " RESET_COLOR, j);

		if (showWeights == true)
		{
			for (uint32_t k = 0; k < pNetwork->inputs_size[layer_number]; k++)
			{
				printf(CYAN "w[%d]: " BLUE "%f" RESET_COLOR " | ", k, pNetwork->weights[layer_number][j][k]);
			}
		}

		if (showBiases == true)
		{
			printf(CYAN "b[%d]: " RESET_COLOR, j);
			printf(BLUE "%f" RESET_COLOR, pNetwork->biases[layer_number][j]);
		}

		printf("\n");
	}

	if (showNeuronErrors == true)
	{
		printf(YELLOW "err: ");

		for (uint32_t j = 0; j < pNetwork->outputs_size[layer_number]; j++)
		{
			printf("%f ", pNetwork->errors[layer_number][j]);
		}

		printf("\n" RESET_COLOR);
	}

	if (showOuts == true)
	{
		printf(MAGENTA "out: ");

		for (uint32_t j = 0; j < pNetwork->outputs_size[layer_number]; j++)
		{
			printf("%f ", pNetwork->outputs[layer_number][j]);
		}

		printf("\n" RESET_COLOR);
	}
}


void print_last_layer(NeuralNetwork *const pNetwork, const bool showNeuronErrors, const bool showInput, const bool showWeights, const bool showBiases, const bool showOuts)
{
	print_layer(pNetwork, (pNetwork->layers_count - 1), showNeuronErrors, showInput, showWeights, showBiases, showOuts);
}


void print_network(NeuralNetwork *const pNetwork, const bool showLayersCount, const bool showActivationFunction, const bool showNeuronErrors, const bool showInput, const bool showWeights, const bool showBiases, const bool showOuts)
{
	if (showLayersCount == true)
	{
		printf("layers_count: %d\n", pNetwork->layers_count);
	}

	if (showActivationFunction == true)
	{
			printf("activation: unknown\n");
	}

	for (uint32_t i = 0; i < pNetwork->layers_count; i++)
	{
		print_layer(pNetwork, i, showNeuronErrors, showInput, showWeights, showBiases, showOuts);
	}
}
