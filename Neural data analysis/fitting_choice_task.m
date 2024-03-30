load("firing_rates_choice_task.mat")

firing_rates = choice_task.';

block_length = length(firing_rates);

fun1 = @(params) L_power_func(params(1:3), params(4), firing_rates);
fun2 = @(params) L_exponential(params(1:3), params(4), firing_rates);
fun3 = @(params) L_inv_sqrt(params(1:2), params(3), firing_rates);

% fun4 = @(params) L_power_func_no_constant(params(1:2), params(3), avg_responses);

params_guess = [2.2, 2.1, -0.7, 0.1];

options = optimset('MaxFunEvals', 10000);

[params1, loss_min1] = fminsearch(fun1, params_guess, options);
[params2, loss_min2] = fminsearch(fun2, [2.5, 1.5, -0.5, .1], options);
[params3, loss_min3] = fminsearch(fun3, [2, 2, .1], options);
% [params4, loss_min4] = fminsearch(fun4, [0, 0, 1], options);


trial_axis_fine = 1:0.1:block_length;

power_fitted = params1(1) * ones(1, length(trial_axis_fine)) + params1(2) * (trial_axis_fine .^ (params1(3) * ones(1, length(trial_axis_fine))));
inv_sqrt_fitted = params3(1) * ones(1, length(trial_axis_fine)) + params3(2) * (trial_axis_fine .^ (-1/2 * ones(1, length(trial_axis_fine))));
exponential_fitted = params2(1) * ones(1, length(trial_axis_fine)) + params2(2) * (exp(params2(3) * trial_axis_fine));