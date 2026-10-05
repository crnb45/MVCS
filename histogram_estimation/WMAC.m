function [b] = WMAC(W, l, sigma_h, sigma_z)
    %RAYLEIGH_CHANNEL Computes rx signal based on a Rayleigh channel
    % INPUT:
    % W = transmitted signals (real-valued 2T by n matrix)
    % l = path loss (real-valued n-length vector)
    % sigma_h = Rayleigh fading coefficient
    % sigma_z = AWGN coefficient
    % OUTPUT:
    % b = received majority votes ({-1,1}-values T by 1 vector)

    [T2, ~] = size(W);
    T = T2/2;
    
    % Generate fading coefficients
    H = sigma_h*randn(size(W)) + 1j*sigma_h*randn(size(W));
    z = sigma_z*( randn(2*T, 1) + 1j*randn(2*T, 1) );
    
    % W-MAC
    y = sum( ((W.*H)'.*l)' , 2) + z;

    % Collect energy
    y_neg_en = abs(y(1:T)).^2;
    y_pos_en = abs(y(T+1:2*T)).^2;

    % Majority vote
    b = (y_pos_en >= y_neg_en)*2-1;
end

