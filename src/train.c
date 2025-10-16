#include "train.h"


double relu_derivative(const double output)
{
	return output > 0 ? 1 : 0;
}

double sigmoid_derivative(const double output)
{
	return output * (1.0 - output);
}


void back_propagation(NeuralNetwork *const pNetwork, const double *const expected)
{
	for (int i = (pNetwork->layers_count - 1); i >= 0; i--)
	{
		if (i != (pNetwork->layers_count - 1))
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
			}
		}
		else
		{
			for (uint32_t j = 0; j < pNetwork->outputs_size[i]; j++)
			{
				pNetwork->errors[i][j] = expected[j] - pNetwork->outputs[i][j];
			}
		}

		for (uint32_t j = 0; j < pNetwork->outputs_size[i]; j++)
		{
			pNetwork->errors[i][j] *= sigmoid_derivative(pNetwork->outputs[i][j]);
		}
	}
}


void update_weights(NeuralNetwork *const pNetwork, const double *const X, const double learning_rate)
{
	for (int i = 0; i < pNetwork->layers_count; i++)
	{
		if (i != 0)
		{
			for (uint32_t j = 0; j < pNetwork->outputs_size[i]; j++)
			{
				for (uint32_t k = 0; k < pNetwork->inputs_size[i]; k++)
				{
					pNetwork->weights[i][j][k] += learning_rate * pNetwork->errors[i][j] * pNetwork->outputs[i - 1][k];
				}
			}
		}
		else
		{
			for (uint32_t j = 0; j < pNetwork->outputs_size[i]; j++)
			{
				for (uint32_t k = 0; k < pNetwork->inputs_size[i]; k++)
				{
					pNetwork->weights[i][j][k] += learning_rate * pNetwork->errors[i][j] * X[k];
				}
			}
		}

		for (uint32_t j = 0; j < pNetwork->outputs_size[i]; j++)
		{
			pNetwork->biases[i][j] += learning_rate * pNetwork->errors[i][j];
		}
	}
}
