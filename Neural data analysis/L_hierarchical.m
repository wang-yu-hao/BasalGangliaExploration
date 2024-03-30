function L = L_hierarchical(hyper_params, responses, func, random_numbers)

    mu_a = hyper_params(1);
    mu_b = hyper_params(2);
    mu_c = hyper_params(3);

    sigma_a = exp(hyper_params(4));
    sigma_b = exp(hyper_params(5));
    sigma_c = exp(hyper_params(6));

    sigma_y = hyper_params(7);

    % sampling for Monte Carlo
%     sample_size = 1000;
%     a_samples = normrnd(mu_a, sigma_a, sample_size);
%     b_samples = normrnd(mu_b, sigma_b, sample_size);
%     c_samples = normrnd(mu_c, sigma_c, sample_size);
    sample_size = size(random_numbers);
    sample_size = sample_size(2);
    a_samples = mu_a + sigma_a * random_numbers(1, :);
    b_samples = mu_b + sigma_b * random_numbers(2, :);
    c_samples = mu_c + sigma_c * random_numbers(3, :);

    sz = size(responses);
    L = 0;

    for i=1:sz(1)

        for k=1:sample_size

%             if func([a_samples(k), b_samples(k), c_samples(k)], sigma_y, responses(i, 2:end)) == inf
%                 
%                 disp([a_samples(k), b_samples(k), c_samples(k)]);
%                 disp(func([a_samples(k), b_samples(k), c_samples(k)], sigma_y, responses(i, 2:end)));
% 
%             end

            L = L + func([a_samples(k), b_samples(k), c_samples(k)], sigma_y, responses(i, 2:end));

        end

    end

    L = L / sample_size;
    
%     disp(L);








