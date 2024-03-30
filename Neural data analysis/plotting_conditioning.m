% Run entire script (or different section in sequence) as each section adds panel to existing figure.

load('conditioning_results.mat')

%% plotting fitted functions and data

plotted_neurons = [26, 39];

trial_axis = 1:1:sz(2);
trial_axis_fine = 1:0.1:sz(2);

h = figure();

hold off
subplot(3, 2, 1);
errorbar(trial_axis, avg_responses -1, se_responses, 'o', 'DisplayName', "Data", 'LineWidth', 1.5);
hold on
plot(trial_axis_fine, power_fitted-1, 'DisplayName', "Power", 'LineWidth', 1.5);
plot(trial_axis_fine, inv_sqrt_fitted-1, 'DisplayName', "Inv. sq. rt.", 'LineWidth', 1.5);
plot(trial_axis_fine, exponential_fitted-1, 'DisplayName', "Exponential", 'LineWidth', 1.5);

set(gca, 'FontSize', 9)
xlabel("Trials", 'FontSize', 10);
ylabel("Norm response", 'FontSize', 10)
title("Average", 'FontSize', 10);
legend('FontSize', 9);

set(gca, 'box', 'off')

for i = 1:length(plotted_neurons)
    
    hold off
    subplot(3, 2, i+1);
    scatter(trial_axis, firing_rates(plotted_neurons(i),:)-1, 'DisplayName', "Data", 'LineWidth', 1.5);
    hold on
    fitted_power = @(t) power(phi_power_uncorr(:, plotted_neurons(i))' ,t);
    fitted_exp = @(t) exponential(phi_exp_corr(:, plotted_neurons(i))' ,t);
    fitted_invsqrt = @(t) inv_sqrt(phi_invsqrt_corr(:, plotted_neurons(i))' ,t);
    plot(trial_axis_fine, fitted_power(trial_axis_fine)-1, 'DisplayName', "Power", 'LineWidth', 1.5);
    plot(trial_axis_fine, fitted_invsqrt(trial_axis_fine)-1, 'DisplayName', "Inverse square root", 'LineWidth', 1.5);
    plot(trial_axis_fine, fitted_exp(trial_axis_fine)-1, 'DisplayName', "Exponential", 'LineWidth', 1.5);

    set(gca, 'FontSize', 9);
    xlabel("Trials", 'FontSize', 10);
    ylabel("Norm response", 'FontSize', 10);
    title(sprintf("Neuron %d", i), 'FontSize', 10);
end


%% Mean-variance correlation

p = polyfit(avg_responses-1, sd_responses, 1);

subplot(3, 2, i+2);
hold on
ax = gca;
ax.ColorOrderIndex = 5;
scatter(avg_responses-1, sd_responses, 'LineWidth', 1.5);
ax.ColorOrderIndex = 6;
x_axis = linspace(min(avg_responses-1) - 0.1* (max(avg_responses-1) - min(avg_responses-1)), max(avg_responses-1) + 0.1* (max(avg_responses-1) - min(avg_responses-1)));

plot(x_axis, p(1) .* x_axis + p(2), 'LineWidth', 1.5);

xlim([min(avg_responses-1) - 0.1* (max(avg_responses-1) - min(avg_responses-1)), max(avg_responses-1) + 0.1* (max(avg_responses-1) - min(avg_responses-1))]);
ylim([min(sd_responses) - 0.1* (max(sd_responses) - min(sd_responses)), max(p(1) .* x_axis + p(2))]);
xlabel('Mean response', "FontSize", 10);
ylabel('Response s.d.', "FontSize", 10);
dim = [0.865, 0.43, .2, .2];
title("s.d.–mean correlation", 'FontSize', 10)
set(gca, 'FontSize', 9);

% disp(corrcoef(avg_responses-1, sd_responses));
% disp(p);

%% Plotting BIC's

% Avg

subplot(3, 3, 7);

bic1 = 4 * log(30) - 2 * (-loss_min1);
bic2 = 4 * log(30) - 2 * (-loss_min2);
bic3 = 3 * log(30) - 2 * (-loss_min3);

X = categorical({'Power','Inv. sq. rt.','Exponential'});
X = reordercats(X,{'Power','Inv. sq. rt.','Exponential'});


bar(X, [bic1, bic3, bic2], 'FaceColor', [0.4660 0.6740 0.1880]);
set(gca, 'FontSize', 9);
xlabel("Function type", "FontSize", 10);
ylabel("BIC", "FontSize", 10);
title("Average", 'FontSize', 10)

set(gca, 'box', 'off')

% Hierarchical

subplot(3, 3, 8);
bic1 = stats_invsqrt_corr.bic;
bic2 = stats_power_corr.bic;
bic3 = stats_exp_corr.bic;
bic4 = stats_invsqrt_uncorr.bic;
bic5 = stats_power_uncorr.bic;
bic6 = stats_exp_uncorr.bic;


X = categorical({'Power','Inv. sq. rt.','Exponential'});
X = reordercats(X,{'Power','Inv. sq. rt.','Exponential'});

b=bar(X, [bic2, bic5; bic1, bic4; bic3, bic6] );
b(2).FaceColor = 'b';
b(1).FaceColor = [0.6350 0.0780 0.1840];
set(gca, 'FontSize', 9);
legend(["Correlated", "Uncorrelated"], 'FontSize', 9, 'Location', 'northwest');
xlabel("Function type", "FontSize", 10);
ylabel("BIC", "FontSize", 10);
title("Hierarchical", 'FontSize', 10)
ylim([6670, 6710]);

set(gca, 'box', 'off')


%% Plot histogram of power parameter p

subplot(3, 3, 9);
ax.ColorOrderIndex = 9;
histogram(phi_power_uncorr(3, :), 10, 'FaceColor', 'k', 'DisplayName', "Hierarchical");
hold on
plot([params1(3), params1(3)], [0,15], '--', 'Color', 'r', 'DisplayName', "Average", 'LineWidth', 1.5)
set(gca, 'FontSize', 9);
xlabel("Power parameter p", "FontSize", 10);
ylabel("Number of neurons", "FontSize", 10);
legend('FontSize', 9, 'Location', 'northeast');
ylim([0,20]);

set(gca, 'box', 'off')

h.Position(3:4) = [800, 800];
set(h,'Units','Inches');
pos = get(h,'Position');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(h,'figures/lak_fitting_all_updated_layout','-dpdf','-r0')