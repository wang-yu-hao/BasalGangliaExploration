function [L, Q, S] = L_ucb(params, dt_subject)

%     alpha_q = alpha * (1 + epsilon)/2;
%     alpha_s = alpha * (1 - epsilon)/2;

    % Range-limited parameters
%     logit_alpha_q = params(1);
%     logit_alpha_s = params(2);
%     log_beta_1 = params(3); 
%     log_beta_2 = params(4); 
%  
%     
%     alpha_q = 1 / (1 + exp(-logit_alpha_q));
%     alpha_s = 1 / (1 + exp(-logit_alpha_s));
%     beta_1 = exp(log_beta_1);
%     beta_2 = exp(log_beta_2);

    % Raw parameters
%     logit_alpha_q = params(1);
%     logit_alpha_s = params(2);
%     alpha_q = 1 / (1 + exp(-logit_alpha_q));
%     alpha_s = 1 / (1 + exp(-logit_alpha_s));
    alpha_q = params(1);
    alpha_s = params(2);
    beta_1 = params(3);
    beta_2 = params(4);
    

    L = 0;

    Q = [];
    S = [];

    if dt_subject.N <= 200
        num_arms = 2;        
%         s0 = sqrt(10);
        s0 = 10;
        q0 = 0;
    else
        num_arms = 4;        
        s0 = sqrt(12.5);
        q0 = 50;  
    end

    for i=1:max(dt_subject.block)

        q = q0 * ones(1, num_arms);
        counts = zeros(1, num_arms);
        s = s0 * ones(1, num_arms);
%         dt_block = dt_subject(dt_subject.block == i, :);

%         stim1 = dt_block.stim1;
%         stim2 = dt_block.stim2;
        choices = dt_subject.c(dt_subject.block == i);
        shown = dt_subject.R([dt_subject.block == i, dt_subject.block == i]);
        shown = reshape(shown, length(shown)/2, 2);
        r = dt_subject.r(dt_subject.block == i);

        for t=1:length(choices)

%             stim1 = dt_block{t, 'stim1'};
%             stim2 = dt_block{t, 'stim2'};
%             choice = dt_block{t, 'stim_chosen'};
%             r = dt_block{t, 'reward'};

%             P = exp(beta * q(choice)) ...
%             / (exp(beta * q(stim1)) + exp(beta * q(stim2)));           
            

            c = counts;
            c(c==0) = .1;
            
            % Logistic
%             A = q + s ./ sqrt(c);
%             P = exp(beta_1 * A(choices(t))) ...
%             / (exp(beta_1 * A(1)) + exp(beta_1 * A(2)));

            % Individual weights for q and s
            A = beta_1 * q + beta_2 * (s .* (c.^ (-0.5)));
            if num_arms == 2
%                 P = 1 / (1 + exp(-((2*A(choices(t)) - sum(A)))));
                P = normcdf(2*A(choices(t)) - sum(A));
            else
 %                 P = 1 / (1 + exp(-((2*A(choices(t)) - A(shown(t, 1)) - A(shown(t, 2))))));
                P = normcdf(2*A(choices(t)) - A(shown(t, 1)) - A(shown(t, 2)));
            end
            % Probit with individual weights for q and s
%             A = beta_1 * q + beta_2 * (s ./ sqrt(c));
%             P = normcdf(2 * A(choices(t)) - sum(A));

%             disp(P);

            qq(t, :) = q;
            c = counts;
            c(c==0) = 1;
            ss(t, :) = s ./ sqrt(c);
            
            s(choices(t)) = s(choices(t)) + alpha_s * (abs(r(t) - q(choices(t))) - s(choices(t)));
            q(choices(t)) = q(choices(t)) + alpha_q * (r(t) - q(choices(t)));
%             disp(q);
            counts(choices(t)) = counts(choices(t)) + 1;

            L = L - log(P);

        end

        Q = cat(1, Q, qq);
        S = cat(1, S, ss);


%         L = L - log(normpdf(logit_alpha_q, prior_means(1), sqrt(prior_vars(1)))) ...
%             - log(normpdf(log_beta, prior_means(2), sqrt(prior_vars(2))));
 
    end

end