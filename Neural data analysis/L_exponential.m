function L = L_exponential(params, sigma_y, responses)

    a = params(1);
    b = params(2);
    c = params(3);

    block_length = length(responses);
    trial_axis = 1:1:block_length;

    predictions = a * ones(1, block_length) + b * (exp(c * trial_axis));


    noise = predictions - responses;

    L = -sum(log(normpdf(noise, 0, sigma_y)));

end