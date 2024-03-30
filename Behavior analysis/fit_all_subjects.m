function [bic, all_params, latents, all_loss, aic, mean_params, var_params, mean_loss] = fit_all_subjects(loss_fun, dt_all, params_guess, lb, ub, num_guesses, options)

    disp_text = 'Fitting #%d of %d subjects...\n';
    subject_count = length(dt_all);
    params_count = length(params_guess);
    all_params = zeros(subject_count, params_count);
    % all_loss = zeros(1, subject_count);

    params_guess = params_guess.';
    lb = lb.';
    ub = ub.';

    % Constraints for alpha_s < alpha_q (set both to [] to remove)
    if params_count > 2
        A = zeros(params_count);
        A(1, 1) = -1;
        A(1, 2) = 1;
        b = zeros(params_count, 1);
    else
        A = [];
        b = [];
    end

    for n = 1:num_guesses

        disp(n);
        disp(loss_fun);

        for k=1:subject_count
    
            dt_subject = dt_all(k);
    
            fun = @(params) loss_fun(params, dt_subject);

            if n == 1
                guess = params_guess;
                if isa(options, 'string')
                    [params, loss_min] = ga(fun, length(guess), A, b, [], [], lb, ub);
                else
                    [params, loss_min] = fmincon(fun, guess, A, b, [], [], lb, ub, [], options);
                end
            else
                while true
                    guess = normrnd(mean_params.', 2 * sqrt(var_params.'));
                    guess = min(guess, ub);
                    guess = max(guess, lb);
                    try
                        if isa(options, 'string')
                            [params, loss_min] = ga(fun, length(guess), A, b, [], [], lb, ub);
                        else
                            [params, loss_min] = fmincon(fun, guess, A, b, [], [], lb, ub, [], options);
                        end
                        break
                    catch
                    end
%                     if ~isnan(fun(guess)) && fun(guess)~=Inf && fun(guess)~=-Inf
%                         break
%                     end
                end
            end

%             disp(guess);
    
            [L, Q, S] = loss_fun(params, dt_subject);
    
            latents_temp(k).m = Q;
            latents_temp(k).s = S.^2;
    
            all_params_temp(k, :) = params.';
            all_loss_temp(k, 1) = loss_min;
            bic_temp(k, 1) = 2 * loss_min + params_count * log(length(dt_subject.c));
            aic_temp(k, 1) = 2 * loss_min + params_count * 2;
    
        end
        
        if n == 1 || mean(all_loss_temp) < mean_loss
            all_params_temp_trimmed = all_params_temp(all(abs(all_params_temp) < 100, 2), :);
            mean_params = mean(all_params_temp_trimmed);
            var_params = var(all_params_temp_trimmed);
            mean_loss = mean(all_loss_temp);
            latents = latents_temp;
            all_params = all_params_temp;
            all_loss = all_loss_temp;
            bic = bic_temp;
            aic = aic_temp;
        end
%         disp(mean_params);
%         disp(var_params);

        disp(sum(bic));

    end
end