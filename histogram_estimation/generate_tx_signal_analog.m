function [W] = generate_tx_signal_analog(M, X, P, l)
    %GENERATE_TX_SIGNAL Generates transmitted signal for n devices based on
    %their items and the measurement matrix
    %INPUT:
    % M = measurement matrix (real-valued T by d matrix)
    % X = one-hot items (logical d by n matrix)
    %OUTPUT:
    % W = transmitted signals (real-valued 2T by n matrix)

    % Extract constants
        [T, ~] = size(M);
        [~, n] = size(X);
    
        % Compute measurement
        S = M*X;
    
        % Power control factor
        p = sqrt(2*pi)*P*min(l)^2./(l.^2);
    
        % Init votes
        W = zeros(2*T, n);
    
        % Negative votes
        W(1:T, :) = sqrt( abs(S').*p )'.*heaviside(-S);

        % Positive votes
        W(T+1:2*T, :) = sqrt( abs(S').*p )'.*heaviside(S);

end

