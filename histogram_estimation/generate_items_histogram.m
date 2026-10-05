function [X, x_star] = generate_items_histogram(d, n, seed, bins)
    %GENERATE_ITEMS_HISTOGRAM Generates the one-hot items to be used for the
    %ground-truth in the histogram estimation problem.
    % INPUT:
    % d = dimension of items (real-valued scalar)
    % n = number of items (real-valued scalar)
    % seed = seed for random generation (non-negative integer)
    % bins = upper bound of number of nonempty bins in histogram.
    %           if larger than n, sample each item from uniform[1,d]
    %           (non-negative integer)
    % OUTPUT:
    % X = matrix containing n messages (d by n real matrix)
    % x = target vector
    
    rng(seed(1))
    X = zeros(d, n);

    if bins > n
        for i = 1:n
            % Select non-zero index
            nonzero_idx = randi(d, 1);
            X(nonzero_idx, i) = X(nonzero_idx, i) + 1;
        end
    else
        nonzero_loc = randperm(d, bins);   % candidates of nonzero locations of x_star
        for i = 1:n
            % each user selects one of nonzero_loc as its nonzero index
            idx = randi(bins);
            nonzero_idx = nonzero_loc(idx);
            X(nonzero_idx, i) = X(nonzero_idx, i) + 1;
        end
    end
    

    x_star = sum(X, 2);
end

