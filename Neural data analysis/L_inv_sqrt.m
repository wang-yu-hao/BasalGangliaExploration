function L = L_inv_sqrt(params, sigma_y, responses)

    a = params(1);
    b = params(2);

    block_length = length(responses);
    trial_axis = 1:1:block_length;

    predictions = a * ones(1, block_length) + b * (trial_axis .^ (- ones(1, block_length) / 2));


    noise = predictions - responses;

    L = -sum(log(normpdf(noise, 0, sigma_y)));

end