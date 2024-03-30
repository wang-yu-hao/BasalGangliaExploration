function [bms_results, BIC, params, latents, LOSS, AIC] = model_comparison(k)

    if isa(k, "double")
        for i = k % only using one specified experiment
            if i==1
                load results_sim1
                data = load_data(1);
                param = [10 10 0];
            elseif i==2
                load results_sim2
                data = load_data(2);
                param = [10 100 100];
            elseif i==3
                data = load_data(3);
                param = [100 100];
            elseif i==4
                data = load_data(4);
                param = [100 100];
            end
            
            for s = 1:length(data)
                latents(s) = kalman_filter(param,data(s));
            end
            
            for j = 1:4
    
                if size(latents(1).m, 2) == 2
    
                    for s = 1:length(data)
                        X = [];  
                        for n = 1:length(data(s).c)
                            if j==1 % UCB
                                X(n,:) = [latents(s).m(n,1)-latents(s).m(n,2) sqrt(latents(s).s(n,1))-sqrt(latents(s).s(n,2))];
                            elseif j==2 % Thompson
                                X(n,1) = (latents(s).m(n,1)-latents(s).m(n,2))./sqrt(sum(latents(s).s(n,:)));
                            elseif j==3 % Hybrid
                                X(n,:) = [(latents(s).m(n,1)-latents(s).m(n,2))./sqrt(sum(latents(s).s(n,:))) sqrt(latents(s).s(n,1))-sqrt(latents(s).s(n,2))];
                            elseif j==4 % Value-directed exploration
                                X(n,:) = latents(s).m(n,1)-latents(s).m(n,2);
                            end
                        end
                        c = data(s).c;
                        b = glmfit(X,c==1,'binomial','link','probit','constant','off');
                        params(j).w(s, :) = b.'; 
                        y = glmval(b,X,'probit','constant','off');
                        L = sum(log(y(c==1))) + sum(log(1-y(c==2)));
                        loss(s,j) = -L;
                        bic(s,j) = -2*L + size(X,2)*log(n);
                        aic(s,j) = -2*L + 2*size(X,2);
                    end            
                else
                    for s = 1:length(data)
                        X = [];  
                        for n = 1:length(data(s).c)
                            if j==1 % UCB
                                X(n,:) = [latents(s).m(n,data(s).R(n,1))-latents(s).m(n,data(s).R(n,2)) sqrt(latents(s).s(n,data(s).R(n,1)))-sqrt(latents(s).s(n,data(s).R(n,2)))];
                            elseif j==2 % Thompson
                                X(n,1) = (latents(s).m(n,data(s).R(n,1))-latents(s).m(n,data(s).R(n,2)))./sqrt(latents(s).s(n,data(s).R(n,1)) + latents(s).s(n,data(s).R(n,2)));
                            elseif j==3 % Hybrid
                                X(n,:) = [(latents(s).m(n,data(s).R(n,1))-latents(s).m(n,data(s).R(n,2)))./sqrt(latents(s).s(n,data(s).R(n,1)) + latents(s).s(n,data(s).R(n,2))) sqrt(latents(s).s(n,data(s).R(n,1)))-sqrt(latents(s).s(n,data(s).R(n,2)))];
                            elseif j==4 % Value-directed exploration
                                X(n,:) = latents(s).m(n,data(s).R(n,1))-latents(s).m(n,data(s).R(n,2));
                            end
                        end
                        c = data(s).c;
                        shown = data(s).R;
                        b = glmfit(X,c==shown(:, 1),'binomial','link','probit','constant','off');
                        params(j).w(s, :) = b.'; 
                        y = glmval(b,X,'probit','constant','off');
                        L = sum(log(y(c==shown(:, 1)))) + sum(log(1-y(c==shown(:, 2))));
                        loss(s,j) = -L;
                        bic(s,j) = -2*L + size(X,2)*log(n);
                        aic(s,j) = -2*L + 2*size(X,2);
                    end
    
                end
    
    
            end
            
            [bms_results(i).alpha,bms_results(i).exp_r,bms_results(i).xp,bms_results(i).pxp,bms_results(i).bor] = bms(-0.5*bic);
            BIC(i).bic = bic;
            LOSS(i).loss = loss;
            AIC(i).aic = aic;
    
            clear bic
            clear loss
            clear aic
        end

    else
        data = k{2};
        k = k{1};

        for i = k % only using one specified experiment
            if i==1
                load results_sim1
                param = [10 10 0];
    %             param = [0, 10];
            elseif i==2
                load results_sim2
                param = [10 100 100];
    %             param = [0, 10];
            elseif i==3
                param = [156 100];
            elseif i==4
                param = [156 100];
            end
            
            for s = 1:length(data)
                latents(s) = kalman_filter(param,data(s));
            end
            
            for j = 1:4
    
                if size(latents(1).m, 2) == 2
    
                    for s = 1:length(data)
                        X = [];  
                        for n = 1:length(data(s).c)
                            if j==1 % UCB
                                X(n,:) = [latents(s).m(n,1)-latents(s).m(n,2) sqrt(latents(s).s(n,1))-sqrt(latents(s).s(n,2))];
                            elseif j==2 % Thompson
                                X(n,1) = (latents(s).m(n,1)-latents(s).m(n,2))./sqrt(sum(latents(s).s(n,:)));
                            elseif j==3 % Hybrid
                                X(n,:) = [(latents(s).m(n,1)-latents(s).m(n,2))./sqrt(sum(latents(s).s(n,:))) sqrt(latents(s).s(n,1))-sqrt(latents(s).s(n,2))];
                            elseif j==4 % Value-directed exploration
                                X(n,:) = latents(s).m(n,1)-latents(s).m(n,2);
                            end
                        end
                        c = data(s).c;
                        b = glmfit(X,c==1,'binomial','link','probit','constant','off');
                        params(j).w(s, :) = b.'; 
                        y = glmval(b,X,'probit','constant','off');
                        L = sum(log(y(c==1))) + sum(log(1-y(c==2)));
                        bic(s,j) = -2*L + size(X,2)*log(n);
                        loss(s,j) = -L;
                        aic(s,j) = -2*L + 2*size(X,2);
                    end            
                else
                    for s = 1:length(data)
                        X = [];  
                        for n = 1:length(data(s).c)
                            if j==1 % UCB
                                X(n,:) = [latents(s).m(n,data(s).R(n,1))-latents(s).m(n,data(s).R(n,2)) sqrt(latents(s).s(n,data(s).R(n,1)))-sqrt(latents(s).s(n,data(s).R(n,2)))];
                            elseif j==2 % Thompson
                                X(n,1) = (latents(s).m(n,data(s).R(n,1))-latents(s).m(n,data(s).R(n,2)))./sqrt(latents(s).s(n,data(s).R(n,1)) + latents(s).s(n,data(s).R(n,2)));
                            elseif j==3 % Hybrid
                                X(n,:) = [(latents(s).m(n,data(s).R(n,1))-latents(s).m(n,data(s).R(n,2)))./sqrt(latents(s).s(n,data(s).R(n,1)) + latents(s).s(n,data(s).R(n,2))) sqrt(latents(s).s(n,data(s).R(n,1)))-sqrt(latents(s).s(n,data(s).R(n,2)))];
                            elseif j==4 % Value-directed exploration
                                X(n,:) = latents(s).m(n,data(s).R(n,1))-latents(s).m(n,data(s).R(n,2));
                            end
                        end
                        c = data(s).c;
                        shown = data(s).R;
                        b = glmfit(X,c==shown(:, 1),'binomial','link','probit','constant','off');
                        params(j).w(s, :) = b.'; 
                        y = glmval(b,X,'probit','constant','off');
                        L = sum(log(y(c==shown(:, 1)))) + sum(log(1-y(c==shown(:, 2))));
                        bic(s,j) = -2*L + size(X,2)*log(n);
                        loss(s,j) = -L;
                        aic(s,j) = -2*L + 2*size(X,2);
                    end
    
                end
    
    
            end
            
            [bms_results(i).alpha,bms_results(i).exp_r,bms_results(i).xp,bms_results(i).pxp,bms_results(i).bor] = bms(-0.5*bic);
            BIC(i).bic = bic;
            LOSS(i).loss = loss;
            AIC(i).aic = aic;
    
            clear bic
            clear loss
            clear aic
        end

    end