function [x_hat] = passive_reconstruct(gamma0, a, d, n)
    %PASSIVE_RECONSTRUCT Runs the passive 1-bit CS reconstruction alg
    % INPUT:
    % gamma0 = scalar hyperparameter (just gamma in Zhang's paper)
    % a = d-length vector containing the argument for the soft thresholding
    % d = scalar dimension of the histogram
    % n = number of user devices

    % Passive reconstruction algorithm
    if max(a) <= gamma0
        x_hat = zeros(d, 1);
    else
        a = soft_thresholding(a, gamma0);
        x_hat = a/norm(a, 2);
    end
end

