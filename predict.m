function p = predict(Theta1, Theta2, X)
% Predict the label class of an input digit given a trained neural network

% Number of training examples and classes
m = size(X, 1);
num_labels = size(Theta2, 1);

% The predicted class matrix 
p = zeros(size(X, 1), 1);

h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');
[dummy, p] = max(h2, [], 2);

end
