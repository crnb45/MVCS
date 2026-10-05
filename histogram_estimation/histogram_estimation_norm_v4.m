clear
clc

%Params
sigma_h = 1;
sigma_z = 0.1;
d = 10000;                %dimension of histogram
n = 10;                %number of devices
R = 2;                  %Radius of cell
P = 1;                  %Transmission power
beta = 2;               %Path loss coefficient
delta = 3;              %See Theorem 1
c_iters = 100;           %Number of Monte-Carlo simulations
bins = n+1;   %For overlap between user items

A = n + n + 0.5*sqrt(n);
%If epsilon goes above this threshold we mark that as an error
epsilon_thresh = min(0.5/n, sqrt(n)/(A + sqrt(A^2 - 2 * n * sqrt(n))));

%If we don't want to optimize, select good_c
good_c=0.4;
% d=500, n=5, c=0.12

% calculate analytical bound for T using good_c
E = power(2*pi, 1/4) * sqrt(P) * power(R, -beta/2);
mult = 2*sigma_z^2 / (E*sigma_h)^2;
lambda = sqrt(n) / (sqrt(2/pi) * (sqrt(n)+n) + mult);
analytical_T = (delta + log(d)) * (6*good_c / lambda / epsilon_thresh)^2 * n    % sparsity of x_star <= n

%%
T_list = 500:500:6000;    %Number of channel uses
len_T = length(T_list);

%Common random seed
seeds = randi(power(2,30), c_iters, 1); 

%For each list, first column corresponds to '1-norm' case (line 78~81)
%and second column corresponds to 'sum' case (line 83~86)
%Epsilon list
error_list_T = zeros(len_T, 2);
%Probability of exceeding epsilon_thresh list
prob_tresh = zeros(len_T, 2);
%Probability of exactly estimating the histogram
exact_list = zeros(len_T, 2);

tic
%Run the whole generation+measurement+communication+reconstruction
for T_i = 1:len_T
    T = T_list(T_i);

    current_time = datetime('now', 'Format','HH:mm:ss');
    current_time = string(current_time);
    time_log = append('[', current_time, '] ', string(T_i), '/', string(len_T), ', T=', string(T));
    disp(time_log)
    
    cumulative_epsilon_sum = 0;
    num_thresh_sum = 0;
    exact_sum = 0;

    % cumulative_epsilon_norm = 0;
    % num_thresh_norm = 0;
    % exact_norm = 0;

    for c_i = 1:c_iters
        %Generate items
        [X, x_star] = generate_items_histogram(d, n, seeds(c_i), bins);
        
        %Measure
        M = randn(T, d);

        %Generate path loss
        l = generate_pathloss(n, R, beta);
    
        % Transmit        
        W = generate_tx_signal_analog(M, X, P, l);
    
        % Receive
        b = WMAC(W, l, sigma_h, sigma_z);

        b_star = sign(M*x_star);

        %Passive reconstruction
        a = 1/T*(M')*b;
        
        c = good_c;
        gamma0 = 2*c*sqrt((delta+log(d))/T);
        x_hat_original = passive_reconstruct(gamma0, a, d, n);
        % sparsity = nnz(x_hat);
        % disp(x_hat(1:10))
        % disp(sparsity)

        % ***************** threshold variation *****************
        % Find the threshold value 
        threshold = 0.5/n;
        
        % Keep elements greater than or equal to the threshold, set others to zero
        x_hat = x_hat_original .* (x_hat_original >= threshold);
        % *******************************************************

        % %Get exact histogram using 1-norm of x_hat
        % x_hat_norm = x_hat / norm(x_hat, 1) * n;
        % x_hat_norm_round = round(x_hat_norm);
        % x_hat_norm_final = clip(x_hat_norm_round, 0, Inf);

        %Get exact histogram using sum of the elements of x_hat
        x_hat_sum = x_hat / sum(x_hat) * n;
        x_hat_sum_round = round(x_hat_sum);
        x_hat_sum_final = clip(x_hat_sum_round, 0, Inf);

        % %1-norm case result
        % epsilon_norm = norm(x_hat - x_star/norm(x_star,2), 2);
        % cumulative_epsilon_norm = cumulative_epsilon_norm + epsilon_norm;
        % if epsilon_norm < epsilon_thresh
        %     num_thresh_norm = num_thresh_norm + 1;
        % end
        % if x_star == x_hat_norm_final
        %     exact_norm = exact_norm + 1;
        % end

        %sum case epsilon
        epsilon_sum = norm(x_hat - x_star/norm(x_star,2), 2);
        cumulative_epsilon_sum = cumulative_epsilon_sum + epsilon_sum;
        if epsilon_sum < epsilon_thresh
            num_thresh_sum = num_thresh_sum + 1;
        end

        %sum case exact histogram estimation
        if x_star == x_hat_sum_final
            exact_sum = exact_sum + 1;
        end

        clear("M")
        clear("W")
    end

    % error_list_T(T_i, 1) = cumulative_epsilon_norm/c_iters;
    % prob_tresh(T_i, 1) = num_thresh_norm/c_iters;
    % exact_list(T_i, 1) = exact_norm/c_iters;

    error_list_T(T_i, 2) = cumulative_epsilon_sum/c_iters;
    prob_tresh(T_i, 2) = num_thresh_sum/c_iters;
    exact_list(T_i, 2) = exact_sum/c_iters;
end
toc

filename = "d=" + num2str(d);
% dump big matrices before saving
clear("M")
clear("W")
% save results
save(filename)

%%
% Plot
sgtitle(['d=', num2str(d), ', n=', num2str(n), ', c=', num2str(c), ', analy-T=', num2str(analytical_T)])

subplot(2, 2, 1);
%plot(T_list, error_list_T(:, 1))
hold on
plot(T_list, error_list_T(:, 2))
xline(analytical_T, 'LineStyle', '--')
hold off
xlim([0 T_list(length(T_list))])
yline(epsilon_thresh, '--r')
xlabel("T")
ylabel("\epsilon")
title("2-norm error")
%legend('norm', 'sum')

subplot(2, 2, 3);
%plot(T_list, prob_tresh(:, 1))
hold on
plot(T_list, prob_tresh(:, 2))
xline(analytical_T, 'LineStyle', '--')
hold off
xlim([0 T_list(length(T_list))])
yline(1-exp(1-delta), '--b')
xlabel("T")
ylabel("Probability")
title("P(2-norm error < epsilon thresh)")
%legend('norm', 'sum')

subplot(2, 2, 4);
%plot(T_list, exact_list(:, 1))
hold on
plot(T_list, exact_list(:, 2))
xline(analytical_T, 'LineStyle', '--')
hold off
xlim([0 T_list(length(T_list))])
yline(1-exp(1-delta), '--b')
xlabel("T")
ylabel("Probability")
title("Exact histogram estimation")
%legend('norm', 'sum')

%% save
% dump big matrices before saving
clear("M")
clear("W")
% save results
save("d5e4_n10_c0.4_new.mat")

writematrix(T_list, 'T_list.xlsx')
writematrix(prob_tresh(:, 2), 'prob_thresh.xlsx')
writematrix(exact_list(:, 2), 'exact_list.xlsx')

% %% open
% load('d500_n5_theoretical_eps.mat')

% %% Plot_figure 1
% sgtitle(['d=', num2str(d), ', n=', num2str(n), ', c=', num2str(c), ', analy-T=', num2str(analytical_T)])
% 
% subplot(2, 2, 1);
% %plot(T_list, error_list_T(:, 1))
% hold on
% plot(T_list, error_list_T(:, 2))
% xline(analytical_T, 'LineStyle', '--')
% hold off
% xlim([0 T_list(length(T_list))])
% yline(epsilon_thresh, '--r')
% xlabel("T")
% ylabel("\epsilon")
% title("2-norm error")
% 
% subplot(2, 2, 3);
% %plot(T_list, prob_tresh(:, 1))
% hold on
% plot(T_list, prob_tresh(:, 2))
% plot(T_list, exact_list(:, 2))
% xline(analytical_T, 'LineStyle', '--')
% hold off
% yline(1-exp(1-delta), '--b')
% xlabel("T")
% ylabel("Probability")
% legend('$\ell_2$ error $\leq \epsilon$', 'exact histogram estimation', 'Interpreter', 'latex')