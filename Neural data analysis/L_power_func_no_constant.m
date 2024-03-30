function L = L_power_func_no_constant(params, sigma_y, responses)


    b = params(1);
    c = params(2);

    block_length = length(responses);
    trial_axis = 1:1:block_length;

    predictions = ones(1, block_length) + b * (trial_axis .^ (c * ones(1, block_length)));

    noise = predictions - responses;

    L = -sum(log(normpdf(noise, 0, sigma_y)));

end