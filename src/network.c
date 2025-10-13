#include "network.h"


void softmax(double* input, double* output, int length)
{
    double sum = 0.0;
    for (int i = 0; i < length; i++) {
        output[i] = exp(input[i]);
        sum += output[i];
    }
    for (int i = 0; i < length; i++) {
        output[i] /= sum;
    }
}


void forward_layer(const double *const pInput, const uint32_t input_size, double *const *const ppWeights, const double *const pBiases, double *const pOutputs, const uint32_t output_size, PFN_activation_callback activation_callback)
{
	for (uint32_t i = 0; i < output_size; i++)
	{
		pOutputs[i] = 0.0;
		for (uint32_t j = 0; j < input_size; j++)
		{
			pOutputs[i] += pInput[j] * ppWeights[i][j];
		}

		pOutputs[i] += pBiases[i];

		activation_callback(&pOutputs[i]);
	}
}


void forward_network(NeuralNetwork *const pNetwork, const double *const pX, const bool random_fill, const bool print)
{
	forward_layer(pX, pNetwork->inputs_size[0], pNetwork->weights[0], pNetwork->biases[0], pNetwork->outputs[0], pNetwork->outputs_size[0], pNetwork->activation_callback);

	for (uint32_t i = 1; i < pNetwork->layers_count; i++)
	{
		forward_layer(pNetwork->outputs[i-1], pNetwork->inputs_size[i], pNetwork->weights[i], pNetwork->biases[i], pNetwork->outputs[i], pNetwork->outputs_size[i], pNetwork->activation_callback);
	}
	//softmax(pNetwork->outputs[pNetwork->layers_count-1], pNetwork->outputs[pNetwork->layers_count-1], pNetwork->outputs_size[pNetwork->layers_count-1]);
}


void init_network(NeuralNetwork *const pNetwork, const uint32_t X_size, const uint32_t layers_count, const uint32_t *const pLayers_output_size, PFN_activation_callback activation_callback)
{
	pNetwork->activation_callback = activation_callback;
	pNetwork->layers_count = layers_count;

	pNetwork->weights = malloc(sizeof(pNetwork->weights) * layers_count);

	pNetwork->biases = malloc(sizeof(pNetwork->biases) * layers_count);
	pNetwork->outputs = malloc(sizeof(pNetwork->outputs) * layers_count);
	pNetwork->errors = malloc(sizeof(pNetwork->errors) * layers_count);

	pNetwork->inputs_size = malloc(sizeof(*pNetwork->inputs_size) * layers_count);
	pNetwork->outputs_size = malloc(sizeof(*pNetwork->outputs_size) * layers_count);

	for (uint32_t i = 0; i < layers_count; i++)
	{
		if (i == 0)
		{
			pNetwork->inputs_size[i] = X_size;
		}
		else
		{
			pNetwork->inputs_size[i] = pLayers_output_size[i-1];
		}

		pNetwork->outputs_size[i] = pLayers_output_size[i];

		pNetwork->weights[i] = malloc(sizeof(*pNetwork->weights) * pNetwork->outputs_size[i]);

		pNetwork->biases[i] = malloc(sizeof(*pNetwork->biases) * pNetwork->outputs_size[i]);
		pNetwork->outputs[i] = malloc(sizeof(*pNetwork->outputs) * pNetwork->outputs_size[i]);
		pNetwork->errors[i] = malloc(sizeof(*pNetwork->errors) * pNetwork->outputs_size[i]);

		for (uint32_t j = 0; j < pNetwork->outputs_size[i]; j++)
		{
			pNetwork->biases[i][j] = (FILL_MIN + (rand() / (RAND_MAX / (FILL_MAX - FILL_MIN))));
			srand(rand());

			pNetwork->weights[i][j] = malloc(sizeof(**pNetwork->weights) * pNetwork->inputs_size[i]);

			for (uint32_t k = 0; k < pNetwork->inputs_size[i]; k++)
			{
				pNetwork->weights[i][j][k] = (FILL_MIN + (rand() / (RAND_MAX / (FILL_MAX - FILL_MIN))));
				srand(rand());
			}
		}
	}
}


void free_network(NeuralNetwork *const pNetwork)
{
	for (uint32_t i = 0; i < pNetwork->layers_count; i++)
	{
		for (uint32_t j = 0; j < pNetwork->outputs_size[i]; j++)
		{
			free(pNetwork->weights[i][j]);
		}

		free(pNetwork->weights[i]);
		free(pNetwork->biases[i]);
		free(pNetwork->outputs[i]);
		free(pNetwork->errors[i]);
	}

	free(pNetwork->weights);
	free(pNetwork->biases);
	free(pNetwork->outputs);
	free(pNetwork->errors);
	free(pNetwork->inputs_size);
	free(pNetwork->outputs_size);
}


void clear_network(NeuralNetwork *const pNetwork)
{
	for (uint32_t i = 0; i < pNetwork->layers_count; i++)
	{
		for (uint32_t j = 0; j < pNetwork->outputs_size[i]; j++)
		{
			pNetwork->weights[i][j] = nullptr;
		}

		pNetwork->weights[i] = nullptr;
		pNetwork->biases[i] = nullptr;
		pNetwork->outputs[i] = nullptr;
		pNetwork->errors[i] = nullptr;
	}

	pNetwork->weights = nullptr;
	pNetwork->biases = nullptr;
	pNetwork->outputs = nullptr;
	pNetwork->errors = nullptr;
	pNetwork->inputs_size = nullptr;
	pNetwork->outputs_size = nullptr;

	pNetwork->activation_callback = nullptr;
	pNetwork->layers_count = 0;
}
