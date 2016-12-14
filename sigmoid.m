function g = sigmoid(z)
% This sigmoid function compute the activation value on each layer

g = 1.0 ./ (1.0 + exp(-z));

end
