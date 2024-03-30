function [L, Q, S] = L_thompson_fixed(params, dt_subject)

    a = 1.0741;
    b = 0.3056;
    a = a+b;

    m = 1.6767;
    m = m-1;
    k = 4.4855;
    power = -0.7909;

    alpha_q = params(1);
    alpha_s = params(2);
    lambda = params(3);
    e = 0;

    N = 1;

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

        choices = dt_subject.c(dt_subject.block == i);
        shown = dt_subject.R([dt_subject.block == i, dt_subject.block == i]);
        shown = reshape(shown, length(shown)/2, 2);
        r = dt_subject.r(dt_subject.block == i);

        for t=1:length(choices)
            

            c = counts;
            c(c==0) = .1;
            sigma = s .* (c.^ power);

            if num_arms == 2

                A = (q) ./ sqrt(2 * e^2 + sum(lambda ^2 * s.^2 .* (a + b* (m + k * c .^ power)) .^2) /N);

                P = normcdf(2*A(choices(t)) - sum(A));
%                 P = 1 / (1 + exp(-(2*A(choices(t)) - sum(A))));
            else
                A = (q) ./ sqrt(2 * e^2 + lambda ^2 * s(shown(t, 1))^2 * (a + b* (m + k * c(shown(t, 1))^power)) ^2 /N + s(shown(t, 2))^2 * (a + b* (m + k * c(shown(t, 2))^(-0.5))) ^2 /N);

                P = normcdf(2*A(choices(t)) - A(shown(t, 1)) - A(shown(t, 2)));
%                 P = 1 / (1 + exp(-(2*A(choices(t)) - A(shown(t, 1)) - A(shown(t, 2)))));

            end

            qq(t, :) = q;
            c = counts;
            c(c==0) = 1;
            ss(t, :) = s ./ sqrt(c);
            
            counts(choices(t)) = counts(choices(t)) + 1;
            s(choices(t)) = s(choices(t)) + alpha_s * (abs(r(t) - q(choices(t))) - s(choices(t)));
            q(choices(t)) = q(choices(t)) + alpha_q * (r(t) - q(choices(t)));

            

            L = L - log(P);

        end

        Q = cat(1, Q, qq);
        S = cat(1, S, ss);

    end



end