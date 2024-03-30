function [L, Q, S] = L_rw(params, dt_subject)

%     alpha_q = alpha * (1 + epsilon)/2;
%     alpha_s = alpha * (1 - epsilon)/2;

    % Range-limited parameters
%     logit_alpha_q = params(1);
%     log_beta = params(2); 
%     
%     alpha_q = 1 / (1 + exp(-logit_alpha_q));
%     beta = exp(log_beta);

    % Raw parameters
%     logit_alpha_q = params(1);
%     alpha_q = 1 / (1 + exp(-logit_alpha_q));
    alpha_q = params(1);
    beta = params(2);

    L = 0;
    Q = [];
    if dt_subject.N <= 200
        num_arms = 2;        
        q0 = 0;    
    else
        num_arms = 4;        
        q0 = 50;  
    end

    for i=1:max(dt_subject.block)

        q = q0 * ones(1, num_arms);

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

            if num_arms == 2
                A = beta * q;
%                 P = 1 / (1 + exp(-((2*A(choices(t)) - sum(A)))));
                P = normcdf(2*A(choices(t)) - sum(A));
            else
                A = beta * q;
%                 P = 1 / (1 + exp(-((2*A(choices(t)) - A(shown(t, 1)) - A(shown(t, 2))))));
                P = normcdf(2*A(choices(t)) - A(shown(t, 1)) - A(shown(t, 2)));
            end

            qq(t, :) = q;

            q(choices(t)) = q(choices(t)) + alpha_q * (r(t) - q(choices(t)));
%             disp(q);
        

            L = L - log(P);

        end

        Q = cat(1, Q, qq);

%         L = L - log(normpdf(logit_alpha_q, prior_means(1), sqrt(prior_vars(1)))) ...
%             - log(normpdf(log_beta, prior_means(2), sqrt(prior_vars(2))));

    end
    S = [];
end