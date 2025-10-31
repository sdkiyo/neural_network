#include <stdio.h>
#include <dlfcn.h>
#include <time.h>

#include "../../include/train.h"
#include "../../include/network.h"
#include "../../include/print_network.h"


int main(const int argc, const char *const *const argv)
{
	printf(TRANSPARENT_COLOR "start." RESET_COLOR "\n");
	srand(time(NULL));


	void* nn_base_lib = dlopen("./lib_nn_base.so", RTLD_LAZY);
	if (nn_base_lib == NULL)
	{
		printf(RED "base error\n" RESET_COLOR);
	}
	void* nn_train_lib = dlopen("./lib_nn_train.so", RTLD_LAZY);
	if (nn_train_lib == NULL)
	{
		printf(RED "train error\n" RESET_COLOR);
	}

	PFN_init_network init_network = dlsym(nn_base_lib, "init_network");
	PFN_forward_network forward_network = dlsym(nn_base_lib, "forward_network");
	PFN_free_network free_network = dlsym(nn_base_lib, "free_network");
	PFN_clear_network clear_network = dlsym(nn_base_lib, "clear_network");

	PFN_print_last_layer print_last_layer = dlsym(nn_base_lib, "print_last_layer");
	PFN_print_network print_network = dlsym(nn_base_lib, "print_network");


	PFN_back_propagation back_propagation = dlsym(nn_train_lib, "back_propagation");
	PFN_update_weights update_weights = dlsym(nn_train_lib, "update_weights");

	PFN_activation_callback activation_sigmoid = dlsym(nn_base_lib, "activation_sigmoid");
	PFN_derivative_callback sigmoid_derivative = dlsym(nn_train_lib, "sigmoid_derivative");
	PFN_loss_callback test_loss = dlsym(nn_train_lib, "test_loss");


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

	init_network(&nn, x_size, layers_count, layers_output, activation_sigmoid, sigmoid_derivative, test_loss);

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
		for (uint32_t k = 0; k < 100'000; k++)
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
