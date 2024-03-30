function [L, Q, S] = L_hybrid_fixed(params, dt_subject)

    a = 1.0741;
    b = 0.3056;
    m = 1.6767;
    a = a+b;
    m = m-1;
    k = 4.4855;
    power = -0.7909;

    alpha_q = params(1);
    alpha_s = params(2);
    lambda = params(3);
    e = params(4);

%     N = params(5);
    N = 1;

    [L, Q, S] = L_fixed(alpha_q, alpha_s, lambda, e, m, k, a, b, power, N, dt_subject);

end