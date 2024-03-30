% Takes a few minutes to run.

load("firing_rates_conditioning.mat")

%% Average
firing_rates = DA_Responses_EarlyPostCS_2Monkeys_NovelCues_Laketal(:, 2:end);
sz = size(firing_rates);
avg_responses = mean(firing_rates);
sd_responses = std(firing_rates);
se_responses = sd_responses / sqrt(sz(1));
block_length = length(avg_responses);

fun1 = @(params) L_power_func(params(1:3), params(4), avg_responses);
fun2 = @(params) L_exponential(params(1:3), params(4), avg_responses);
fun3 = @(params) L_inv_sqrt(params(1:2), params(3), avg_responses);


params_guess = [0, 0, 0, 0.5];

options = optimset('MaxFunEvals', 10000);

[params1, loss_min1] = fminsearch(fun1, params_guess, options);
[params2, loss_min2] = fminsearch(fun2, [2, 6, -0.5, 1], options);
[params3, loss_min3] = fminsearch(fun3, [0, 0, 1], options);


trial_axis_fine = 1:0.1:block_length;

power_fitted = params1(1) * ones(1, length(trial_axis_fine)) + params1(2) * (trial_axis_fine .^ (params1(3) * ones(1, length(trial_axis_fine))));
inv_sqrt_fitted = params3(1) * ones(1, length(trial_axis_fine)) + params3(2) * (trial_axis_fine .^ (-1/2 * ones(1, length(trial_axis_fine))));
exponential_fitted = params2(1) * ones(1, length(trial_axis_fine)) + params2(2) * (exp(params2(3) * trial_axis_fine));


%% Hierarchical using nlmefit

firing_rates = DA_Responses_EarlyPostCS_2Monkeys_NovelCues_Laketal(:, 2:end);
sz = size(firing_rates);
trial = 1:sz(2);

% candidate models
power = @(phi, t) phi(1) + phi(2) * (t .^ phi(3));
exponential = @(phi, t) phi(1) + phi(2) * exp(phi(3) * t);
inv_sqrt = @(phi, t) phi(1) + phi(2) * (t .^ (-0.5));


TRIAL = repmat(trial, sz(1), 1);
group = repmat((1:sz(1))', size(trial));

options = statset('MaxIter', 20000);

% fit inverse square root function

beta0 = [0, 0];
[beta_invsqrt_uncorr, psi_invsqrt_uncorr, stats_invsqrt_uncorr, b_invsqrt_uncorr] = nlmefit(TRIAL(:), ...
    firing_rates(:), group(:), [], inv_sqrt, beta0, 'Options', options);
[beta_invsqrt_corr, psi_invsqrt_corr, stats_invsqrt_corr, b_invsqrt_corr] = nlmefitsa(TRIAL(:), ...
    firing_rates(:), group(:), [], inv_sqrt, beta0, 'CovPattern', ones([2, 2]), 'LogLikMethod', 'is', 'ComputeStdErrors', true, 'Options', options);
phi_invsqrt_corr = repmat(beta_invsqrt_corr, 1, sz(1)) + b_invsqrt_corr;
phi_invsqrt_uncorr = repmat(beta_invsqrt_uncorr, 1, sz(1)) + b_invsqrt_uncorr;


% fit power function

beta0 = [0, 0, 0];
[beta_power_uncorr, psi_power_uncorr, stats_power_uncorr, b_power_uncorr] = nlmefitsa(TRIAL(:), ...
    firing_rates(:), group(:), [], power, beta0, 'LogLikMethod', 'is', 'ComputeStdErrors', true, 'Options', options);
[beta_power_corr, psi_power_corr, stats_power_corr, b_power_corr] = nlmefitsa(TRIAL(:), ...
    firing_rates(:), group(:), [], power, beta0, 'CovPattern', ones([3, 3]), 'LogLikMethod', 'is', 'ComputeStdErrors', true, 'Options', options);
phi_power_uncorr = repmat(beta_power_uncorr, 1, sz(1)) + b_power_uncorr;
phi_power_corr = repmat(beta_power_corr, 1, sz(1)) + b_power_corr;

beta0 = [0, 0, 0];
[beta_exp_corr, psi_exp_corr, stats_exp_corr, b_exp_corr] = nlmefitsa(TRIAL(:), ...
    firing_rates(:), group(:), [], exponential, beta0, 'CovPattern', ones([3, 3]), 'LogLikMethod', 'is', 'ComputeStdErrors', true, 'Options', options);
[beta_exp_uncorr, psi_exp_uncorr, stats_exp_uncorr, b_exp_uncorr] = nlmefit(TRIAL(:), ...
    firing_rates(:), group(:), [], exponential, beta0, 'Options', options);
phi_exp_corr = repmat(beta_exp_corr, 1, sz(1)) + b_exp_corr;
phi_exp_uncorr = repmat(beta_exp_uncorr, 1, sz(1)) + b_exp_uncorr;
