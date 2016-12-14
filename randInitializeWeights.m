function W = randInitializeWeights(Layer_in, Layer_out)
% Randomly initialize the weights of parameters theta for the purpose
% of computing more complex and powerful hypothesis function

% Initialize the matrix of parameter theta
W = zeros(Layer_out, 1 + Layer_in);

% Initialize the value of epsilon unit
epsilon_init = 0.12;
% Randomize the weight matrix of theta to break the symmetry
% The first row of W corresponds to the parameters for the bias units
W = rand(Layer_out, 1+Layer_in)*2*epsilon_init-epsilon_init;

end
