#include <math.h>


double relu_derivative(const double output)
{
	return output > 0 ? 1 : 0;
}


double sigmoid_derivative(const double output)
{
	return output * (1.0 - output);
}
