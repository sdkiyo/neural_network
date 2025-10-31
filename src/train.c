#include "train.h"


void back_propagation(NeuralNetwork *const pNetwork, const double *const expected)
{
	for (uint32_t j = 0; j < pNetwork->outputs_size[pNetwork->layers_count - 1]; j++)
	{
		pNetwork->errors[pNetwork->layers_count - 1][j] = pNetwork->loss_callback(pNetwork->outputs[pNetwork->layers_count - 1][j], expected[j]) * pNetwork->derivative_callback(pNetwork->outputs[pNetwork->layers_count - 1][j]);
	}

	for (int i = (pNetwork->layers_count - 2); i >= 0; i--)
	{
		for (uint32_t j = 0; j < pNetwork->outputs_size[i]; j++)
		{
			pNetwork->errors[i][j] = 0.0;

			for (uint32_t k = 0; k < pNetwork->outputs_size[i + 1]; k++)
			{
				for (uint32_t l = 0; l < pNetwork->inputs_size[i + 1]; l++)
				{
					pNetwork->errors[i][j] += pNetwork->weights[i + 1][k][l] * pNetwork->errors[i + 1][k];
				}
			}

			pNetwork->errors[i][j] *= pNetwork->derivative_callback(pNetwork->outputs[i][j]);
		}
	}
}


void update_weights(NeuralNetwork *const pNetwork, const double *const X, const double learning_rate)
{
	for (uint32_t j = 0; j < pNetwork->outputs_size[0]; j++)
	{
		for (uint32_t k = 0; k < pNetwork->inputs_size[0]; k++)
		{
			pNetwork->weights[0][j][k] -= learning_rate * pNetwork->errors[0][j] * X[k];
		}

		pNetwork->biases[0][j] -= learning_rate * pNetwork->errors[0][j];
	}

	for (uint32_t i = 1; i < pNetwork->layers_count; i++)
	{
		for (uint32_t j = 0; j < pNetwork->outputs_size[i]; j++)
		{
			for (uint32_t k = 0; k < pNetwork->inputs_size[i]; k++)
			{
				pNetwork->weights[i][j][k] -= learning_rate * pNetwork->errors[i][j] * pNetwork->outputs[i - 1][k];
			}

			pNetwork->biases[i][j] -= learning_rate * pNetwork->errors[i][j];
		}
	}
}
