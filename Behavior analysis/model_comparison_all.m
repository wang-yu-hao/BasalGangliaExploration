function [bms_results, bic, params, latents, loss, aic] = model_comparison_all(k, n)

    [results, bic1, params, latents, loss1, aic1] = model_comparison(k);

    if isa(k, "double")
        dt = load_data(k);
    else
        dt = k{2};
        k = k{1};
    end

    
    options = optimoptions('fmincon', 'MaxFunctionEvaluations', 20000, 'MaxIterations', 10000, 'Display', 'notify', 'Algorithm', 'interior-point');

    latents = {latents};
    
    for i = 1:4
        
        if i == 4
            [bic2(:, i), p, l, loss2(:, i), aic2(:, i)] = fit_all_subjects(@L_rw, dt, [0.1, 0.2], [0, -Inf], [1, Inf], n, options);
            params(i+4).w = p(:, 2);
            params(i+4).alpha = p(:, 1);
            latents{i+1} = l;
        elseif i == 1
            [bic2(:, i), p, l, loss2(:, i), aic2(:, i)] = fit_all_subjects(@L_ucb, dt, [0.5, 0.5, 0.5, 0.5], [0, 0, -Inf, -Inf], [1, 1, Inf, Inf], n, options);
            params(i+4).w = p(:, 3:4);
            params(i+4).alpha = p(:, 1:2);
            latents{i+1} = l;
        elseif i == 2
            [bic2(:, i), p, l, loss2(:, i), aic2(:, i)] = fit_all_subjects(@L_thompson, dt, [0.05, 0.05, 0.01], [0, 0, -Inf], [1, 1, Inf], n, options);
            params(i+4).w = p(:, 3);
            params(i+4).alpha = p(:, 1:2);
            latents{i+1} = l;
        elseif i == 3
            [bic2(:, i), p, l, loss2(:, i), aic2(:, i)] = fit_all_subjects(@L_hybrid, dt, [0.1, 0.1, 1, 1], [0, 0, -Inf, -Inf], [1, 1, Inf, Inf], n, options);
            params(i+4).w = p(:, 3:4);
            params(i+4).alpha = p(:, 1:2);
            latents{i+1} = l;
        end
        
        clear l p

    end

    for i = 1:4  % dynamic lr
        
        if i == 4
            [bic3(:, i), p, l, loss3(:, i), aic3(:, i)] = fit_all_subjects(@L_rw_dynamic, dt, [1, 10], [0, 0], [inf, inf], n, options);
            params(i+8).w = p(:, 2);
            params(i+8).alpha = p(:, 1);
            latents{i+5} = l;
        elseif i == 1
            [bic3(:, i), p, l, loss3(:, i), aic3(:, i)] = fit_all_subjects(@L_ucb_dynamic, dt, [1, 1, 0.22, 7.5], [0, 0, -inf, 0], [inf, inf, inf, inf], n, options);
            params(i+8).w = p(:, 3:4);
            params(i+8).alpha = p(:, 1:2);
            latents{i+5} = l;
        elseif i == 2
            [bic3(:, i), p, l, loss3(:, i), aic3(:, i)] = fit_all_subjects(@L_thompson_dynamic, dt, [1, 1, 0.2], [0, 0, -Inf], [1, 1, Inf], n, options);
            params(i+8).w = p(:, 3);
            params(i+8).alpha = p(:, 1:2);
            latents{i+5} = l;
        elseif i == 3
            [bic3(:, i), p, l, loss3(:, i), aic3(:, i)] = fit_all_subjects(@L_hybrid_dynamic, dt, [1, 1, 0.2, 7], [0, 0, -inf, 0], [1, 1, inf, inf], n, options);
            params(i+8).w = p(:, 3:4); 
            params(i+8).alpha = p(:, 1:2);
            latents{i+5} = l;
        end
        
        clear l p

    end


    for i = 1:4  % fixed lr
        
        if i == 4
            [bic4(:, i), p, l, loss4(:, i), aic4(:, i)] = fit_all_subjects(@L_rw_fixed, dt, [0.1, 10], [0, 0], [1, inf], n, options);
            params(i+12).w = p(:, 2);
            params(i+12).alpha = p(:, 1);
            latents{i+9} = l;
        elseif i == 1
            [bic4(:, i), p, l, loss4(:, i), aic4(:, i)] = fit_all_subjects(@L_ucb_fixed, dt, [0.05, 0.05, 1, 18], [0, 0, -inf, 0], [1, 1, inf, inf], n, options);
            params(i+12).w = p(:, 3:4);
            params(i+12).alpha = p(:, 1:2);
            latents{i+9} = l;
        elseif i == 2
            [bic4(:, i), p, l, loss4(:, i), aic4(:, i)] = fit_all_subjects(@L_thompson_fixed, dt, [0.1, 0.1, 1], [0, 0, -inf], [1, 1, inf], n, options);
            params(i+12).w = p(:, 3);
            params(i+12).alpha = p(:, 1:2);
            latents{i+9} = l;
        elseif i == 3
            [bic4(:, i), p, l, loss4(:, i), aic4(:, i)] = fit_all_subjects(@L_hybrid_fixed, dt, [.05, .05, 1, 18], [0, 0, -inf, 0], [1, 1, inf, inf], n, options);
            params(i+12).w = p(:, 3:4); 
            params(i+12).alpha = p(:, 1:2);
            latents{i+9} = l;
        end
        
        clear l p

    end


    for i=1:4

        bic = cat(2, bic1(k).bic(:, i), bic2(:, i), bic3(:, i), bic4(:, i));
        % one-to-one-to-one comparison between equivalent Kalman filter and RL and generalised RL based models
        [bms_results(i).alpha,bms_results(i).exp_r,bms_results(i).xp,bms_results(i).pxp,bms_results(i).bor] = bms(-0.5*bic);

    end

    bic = cat(2, bic1(k).bic, bic2, bic3, bic4);
    loss = cat(2, loss1(k).loss, loss2, loss3, loss4);
    aic = cat(2, aic1(k).aic, aic2, aic3, aic4);
    
    % all models
    [bms_results(i+1).alpha,bms_results(i+1).exp_r,bms_results(i+1).xp,bms_results(i+1).pxp,bms_results(i+1).bor] = bms(-0.5* (bic(:, bic(1,:) ~=0)));
    % Kalman filter models
    bms_results(:, i+2) = results(:, k);
    % RL models, mix at behaviour level
    [bms_results(i+3).alpha,bms_results(i+3).exp_r,bms_results(i+3).xp,bms_results(i+3).pxp,bms_results(i+3).bor] = bms(-0.5*bic2);
    % RL models, mix at value level, dynamic learning rates
    [bms_results(i+4).alpha,bms_results(i+4).exp_r,bms_results(i+4).xp,bms_results(i+4).pxp,bms_results(i+4).bor] = bms(-0.5*bic3);
    % RL models, mix at value level, fixed learning rates
    [bms_results(i+5).alpha,bms_results(i+5).exp_r,bms_results(i+5).xp,bms_results(i+5).pxp,bms_results(i+5).bor] = bms(-0.5*bic4);

end


