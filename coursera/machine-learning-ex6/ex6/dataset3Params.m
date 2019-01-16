function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C_train = [0.03, 0.1, 0.3, 1, 3, 10, 30, 0.01]';
%sigma_train = [0.3, 0.1];
sigma_train = [0.1, 0.2, 0.3]';

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
min_last_error = 1;


for i = 1: size(C_train, 1)
   for j = 1: size(sigma_train, 1)
        model = svmTrain(X, y, C_train(i), ...
                        @(x1, x2) gaussianKernel(x1, x2, sigma_train(j)));
        predictions = svmPredict(model, Xval);
        prediction_error(i, j) = mean(double(predictions ~= yval));
        if prediction_error(i, j) < min_last_error
            C = C_train(i);        
            sigma = sigma_train(j);
            min_last_error = prediction_error(i, j);
        end
    end
end

fprintf('sigma es %d y C es %d', sigma, C)


% =========================================================================

end
