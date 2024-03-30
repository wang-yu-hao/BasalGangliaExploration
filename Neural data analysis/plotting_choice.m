load("choice_results.mat")

trial_axis = 1:1:block_length;

h = figure();

hold off
subplot(1, 2, 1);
plot(trial_axis, firing_rates -1, 'o', 'DisplayName', "Data", 'LineWidth', 1.5);
hold on
plot(trial_axis_fine, power_fitted-1, 'DisplayName', "Power", 'LineWidth', 1.5);
plot(trial_axis_fine, inv_sqrt_fitted-1, 'DisplayName', "Inv. sq. rt.", 'LineWidth', 1.5);
plot(trial_axis_fine, exponential_fitted-1, 'DisplayName', "Exponential", 'LineWidth', 1.5);

set(gca, 'FontSize', 9)
xlabel("Trials", 'FontSize', 10);
ylabel("Norm response", 'FontSize', 10)
legend('FontSize', 9);

xlim([0,26]);
ylim([1.2, 3.5]);

subplot(1, 2, 2);
bic1 = 4 * log(block_length) - 2 * (-loss_min1);
bic2 = 4 * log(block_length) - 2 * (-loss_min2);
bic3 = 3 * log(block_length) - 2 * (-loss_min3);

X = categorical({'Power','Inv. sq. rt.','Exponential'});
X = reordercats(X,{'Power','Inv. sq. rt.','Exponential'});

bar(X, [bic1, bic3, bic2]);
set(gca, 'FontSize', 9);
xlabel("Function type", "FontSize", 10);
ylabel("BIC", "FontSize", 10);
ylim([-70, 0]);

h.Position(3:4) = [800, 250];

set(h,'Units','Inches');
pos = get(h,'Position');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(h,'figures/fitting_choice_task','-dpdf','-r0')