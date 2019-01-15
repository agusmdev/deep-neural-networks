function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
%{

h0x = X*theta;

J = (1/(2*m))*((sum(((h0x) - y).**2)) + (lambda/(2*m))*(sum(theta.**2)));



grad(1) = 1 / m * sum((h0x - y) .* X(:, 1));
for i = 2: size(theta, 1)
    grad(i) = 1 / m * sum((h0x - y) .* X(:, i)) + lambda/m * theta(i);
end

%}


J = sum((X * theta - y) .^ 2) / (2 * m);
J += sum(theta(2 : end) .^ 2) * lambda / (2 * m);


p = X' * (X * theta - y) / m;
grad(1) = p(1) ;
grad(2 : end) = p(2 : end) + (lambda / m) * theta(2 : end);


% =========================================================================

grad = grad(:);

end
