function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

% Train the classification of the neural network cost function for 
% one input layer, one hidden layer and one output layer
   
% [J grad]--It returns the cost and gradient of the neural network
% The parameters are unrolled into the vector
% The returned parameter grad should be a "unrolled" vector of the
% partial derivatives of the neural network
                              

% Reshape nn_params back into the parameters Theta1 and Theta2
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Number of training examples
m = size(X, 1);
         
% Initialize the cost value and the partial derivative of cumulative Theta1
% and Theta2
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== Part 1 =========================
% Perform feedforward propagation of the neural network and return the cost
% of variable J. This value'll be verified in the main function.

% Activation matrix of input layer with biased term
a = [ones(m, 1) X];
% Hidden unit matrix in the second layer
features = sigmoid(a*Theta1');
% Adding biased term 1
features = [ones(m, 1) features];
% Output matrix of neural network
predictions = sigmoid(features*Theta2');

% Matrix of indicating if the current training example belongs to class k
Y = zeros(m, num_labels);
for i = 1:m
   Y(i, y(i)) = 1; 
end

YY = zeros(5000,1);
% Cost value due to the hypothesis function
sumValue = (-1)*Y.*log(predictions)-(1-Y).*log(1-predictions);
%Sum over all the output class for each training data.
YY = sum(sumValue, 2); 

% We don't need to consider the biased term of parameters for
% regurlarization
Theta1_unbiased = Theta1(:,2:end);
Theta2_unbiased = Theta2(:,2:end);

% Compute the regurlarization term 
Theta1_value = Theta1_unbiased.^2;
Theta2_value = Theta2_unbiased.^2;

% Compute value of regularized cost function
J = (1/m)*sum(YY)+(lambda/(2*m))*(sum(Theta1_value(:))+sum(Theta2_value(:))); %Cost function value.


% ====================== Part 2 =========================
% Perform backpropagation algorithm of the neural network and return the
% partial derivatives of the cost function with respect to Theta1 and
% Theta2 respectively. The gradient values can be checked by running
% checkNNGradients

% Error matrix in the output layer
delta_3 = predictions - Y;
% Ignore the biased term of Theta2
Theta2_change = Theta2(:,2:end);
% Error matrix in the hidden layer
delta_2 = (delta_3*Theta2_change).*sigmoidGradient(a*Theta1');

% Vectorized implementation of computing the cumulative value of Theta1 and
% Theta2 over all the training examples
Delta_2 = delta_3'*features;
Delta_1 = delta_2'*a;


% ====================== Part 3 =========================
% Implement the regularization with gradient

%For Delta_2
%j!=0 (Excluding the biased term)
Theta2_grad(:,2:end) = (Delta_2(:,2:end)/m)+(lambda/m)*Theta2(:,2:end);
%j==0 (Corresponds to biased term)
Theta2_grad(:,1) = Delta_2(:,1)/m;

%For Delta_1
%j!=0 (Excluding the biased term)
Theta1_grad(:,2:end) = (Delta_1(:,2:end)/m)+(lambda/m)*Theta1(:,2:end);
%j==0 (Corresponds to biased term)
Theta1_grad(:,1) = Delta_1(:,1)/m;


% ====================== Part 4 =========================
% Unroll gradients of Theta1 and Theta2 into vector
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
