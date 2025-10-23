#include "network.h"


void activation_sigmoid(double *const x)
{
	//*x = 1.0 / (1.0 + exp(-*x));
	*x = tanh(*x);
}


void activation_ReLU(double *const x)
{
	if (*x < 0.0)
	{
		*x = 0.0;
	}
}
