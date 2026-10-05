function [l, r] = generate_pathloss(n, R, beta)
    %GENERATE_PATHLOSS Generates pathloss according to uniform circular
    %cell model.
    %INPUT:
    % n = number of devices (positive integer)
    % R = cell radius (positive scalar)
    % beta = path loss exponent (positive scalar)
    %OUTPUT:
    % l = path losses (n-length vector)

    r = zeros(n,1);
    for i = 1:n
        tri = 2; % just some number bigger than 1
        % The distance is a right triangular distribution
        while tri > 1
            r_a = rand(1);
            r_b = rand(1);
            % Sum of two uniform distributions is triangle distribution
            tri = r_a+r_b;
        end
        r(i) = tri*R;
    end
    l = r.^(-beta/2);
end

