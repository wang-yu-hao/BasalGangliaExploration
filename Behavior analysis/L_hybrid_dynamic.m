function [L, Q, S] = L_hybrid_dynamic(params, dt_subject)

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
    e = params(4);

%     N = params(3);
    N = 1;

    [L, Q, S] = L_dynamic(alpha_q, alpha_s, lambda, e, m, k, a, b, power, N, dt_subject);


end