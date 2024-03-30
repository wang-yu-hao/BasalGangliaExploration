function [L, Q, S] = L_rw_dynamic(params, dt_subject)

    a = 0;
    b = 0;
    m = 1.6767;
    k = 4.4855;
    power = -0.7909;

    alpha_q = params(1);
    alpha_s = 0;
    lambda = 0;
    e = params(2);

%     N = params(5);
    N = 1;


    [L, Q, S] = L_dynamic(alpha_q, alpha_s, lambda, e, m, k, a, b, power, N, dt_subject);

end