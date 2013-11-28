function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

a1_with_bias_unit = [ones(rows(X), 1), X];

z2 = a1_with_bias_unit * Theta1';
a2 = sigmoid(z2);
a2_with_bias_unit = [ones(rows(a2), 1), a2];
z3 = a2_with_bias_unit * Theta2';
a3 = sigmoid(z3);

% hypothesis is m x num_lables matrix -- each row corresponds to bitwise y
hypothesis = a3;

cumulative_costs = 0;
% loop through all training examples
for i = 1:m
	hypothesis_row = hypothesis(i,:);

	% create logical array for current y value
	% (1 x num_labelsvector)
	bitwise_y = (1:num_labels) == y(i);

	cost_row = -1 .* bitwise_y .* log(hypothesis_row) .- ((1 .- bitwise_y) .* log(1 .- hypothesis_row));
	single_example_cost = sum(cost_row);

	cumulative_costs += single_example_cost;
end

J = 1 / m * cumulative_costs; 


%regularization
%drop first column
theta_1_no_bias = Theta1;
theta_2_no_bias = Theta2;

theta_1_no_bias(:, [1]) = [];
theta_2_no_bias(:, [1]) = [];

theta_1_no_bias_squared = theta_1_no_bias .^ 2;
theta_2_no_bias_squared = theta_2_no_bias .^ 2;

%sum individual rows, then sum all rows
first_regularization = sum(sum(theta_1_no_bias_squared, 2));
second_regularization = sum(sum(theta_2_no_bias_squared, 2));

regularization_term = lambda / (2 * m) * (first_regularization + second_regularization);

J = J + regularization_term;


%backpropogation algorithm
for t = 1:m
	%column vector for current training example, with bias unit added
	a1_with_bias = [1, X(t,:)];

	z2 = a1_with_bias * Theta1';
		
	a2_with_bias = [1, sigmoid(z2)];
	
	z3 = a2_with_bias * Theta2';
	a3 = sigmoid(z3);

	bitwise_y = (1:num_labels) == y(t);
	delta_3 = a3 - bitwise_y;
	
	delta_3 = delta_3';

	with_bias = Theta2' * delta_3;
	drop_bias = with_bias(2:end);	
	delta_2 = drop_bias .* sigmoidGradient(z2)';

	Theta2_grad = Theta2_grad + delta_3 * a2_with_bias;
	Theta1_grad = Theta1_grad + delta_2 * a1_with_bias;
end

Theta2_grad = Theta2_grad .* (1 / m);
Theta1_grad = Theta1_grad .* (1 / m);

Theta2_grad_reg_adjust = lambda / m .* Theta2;
Theta1_grad_reg_adjust = lambda / m .* Theta1;
Theta2_grad_reg_adjust(:,1) = zeros();
Theta1_grad_reg_adjust(:,1) = zeros();

Theta2_grad = Theta2_grad + Theta2_grad_reg_adjust;
Theta1_grad = Theta1_grad + Theta1_grad_reg_adjust;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
