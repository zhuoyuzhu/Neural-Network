function g = sigmoidGradient(z)
% This function returns the gradient of the sigmoid function evaluated at z
% It'll be used to calculate the error matrix in the Backpropagation 
% algorithm

% Initialize the matirx of sigmoidGradient
g = zeros(size(z));

% Compute the gradient of sigmoid function
z = 1./(1+exp(-z));
g = z.*(1-z);

end
