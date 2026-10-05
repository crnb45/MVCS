function [P] = soft_thresholding(alpha, gamma)
    %SOFT_THRESHOLDING Equation (7) in Zhang2014
    % alpha = real-valued vector of any length (including 1)
    % gamma = real-valued scalar
    P = zeros(size(alpha));
    idx = ( abs(alpha) > gamma );
    P(idx) = sign(alpha(idx)).*(abs(alpha(idx)) - gamma);
end

