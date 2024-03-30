%% run model fitting （takes a few hours)

[bms, bic, params, latents, loss] = model_comparison_all(2, 50);

%% load saved fitting results

load('all_results.mat')

%% bic
bic_sum = sum(bic);
bic_mean = mean(bic);
bic_se = std(bic) / sqrt(size(bic,1));

bic_centred = bic - mean(bic, 2) + mean(bic, "all");
bic_centred_se = std(bic_centred) / sqrt(size(bic_centred,1));


bic_sum_reordered = [bic_sum(3), bic_sum(15); bic_sum(1), bic_sum(13); 
    bic_sum(2), bic_sum(14); bic_sum(4), bic_sum(16)]; % hybrid-ucb-thompson-value, fixed learning rate only
bic_sum_reordered(bic_sum_reordered == 0) = nan;

bic_mean_reordered = bic_sum_reordered / size(bic, 1);


bic_se_reordered = [bic_se(3), bic_se(15); bic_se(1), bic_se(13); 
    bic_se(2), bic_se(14); bic_se(4), bic_se(16)];
bic_se_reordered(bic_se_reordered == 0) = nan;

bic_centred_se_reordered = [bic_centred_se(3), bic_centred_se(15); bic_centred_se(1), bic_centred_se(13); 
    bic_centred_se(2), bic_centred_se(14); bic_centred_se(4), bic_centred_se(16)];
bic_centred_se_reordered(bic_centred_se_reordered == 0) = nan;


X = categorical({'Hybrid','Directed','Random','Value'});
X = reordercats(X,{'Hybrid','Directed','Random','Value'});
h = figure();


b = bar(X, bic_mean_reordered);
ylim([min(bic_mean(bic_mean ~= 0)) - 0.25 * (max(bic_mean) - min(bic_mean(bic_mean ~= 0))), max(bic_mean) + 0.55 * (max(bic_mean) - min(bic_mean(bic_mean ~= 0)))]);

hold on

[ngroups,nbars] = size(bic_mean_reordered);

x = nan(nbars, ngroups);
for i = 1:nbars
    x(i,:) = b(i).XEndPoints;
end

er = errorbar(x', bic_mean_reordered, bic_centred_se_reordered, 'k', 'linestyle', 'none');

set(gca, 'FontSize', 9);

[leg,att] = legendflex(gca, {'Kalman filter', 'BG model'}, 'title', 'Learning rule', 'anchor', {'nw','nw'}, 'buffer', [5 -5]);

set(findall(leg, 'string', 'Learning rule'), 'fontsize', 10);

xlabel("Exploration type", "FontSize", 10);
ylabel("Mean BIC", "FontSize", 10);
h.Position(3:4) = [370, 260];
set(h,'Units','Inches');
pos = get(h,'Position');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);

% print(h,'figures/fitting_compare_manuscript_fixed_only_within_subject_errorbar','-dpdf','-r0')


%% aic

aic_sum = sum(aic);
aic_mean = mean(aic);
aic_se = std(aic) / sqrt(size(aic,1));

aic_centred = aic - mean(aic, 2) + mean(aic, "all");
aic_centred_se = std(aic_centred) / sqrt(size(aic_centred,1));

aic_sum_reordered = [aic_sum(3), aic_sum(15); aic_sum(1), aic_sum(13); 
    aic_sum(2), aic_sum(14); aic_sum(4), aic_sum(16)]; % hybrid-ucb-thompson-value, fixed learning rate only
aic_sum_reordered(aic_sum_reordered == 0) = nan;

aic_mean_reordered = aic_sum_reordered / size(aic, 1);


aic_se_reordered = [aic_se(3), aic_se(15); aic_se(1), aic_se(13); 
    aic_se(2), aic_se(14); aic_se(4), aic_se(16)];
aic_se_reordered(aic_se_reordered == 0) = nan;

aic_centred_se_reordered = [aic_centred_se(3), aic_centred_se(15); aic_centred_se(1), aic_centred_se(13); 
    aic_centred_se(2), aic_centred_se(14); aic_centred_se(4), aic_centred_se(16)];
aic_centred_se_reordered(aic_centred_se_reordered == 0) = nan;


X = categorical({'Hybrid','Directed','Random','Value'});
X = reordercats(X,{'Hybrid','Directed','Random','Value'});
h = figure();

b = bar(X, aic_mean_reordered);
ylim([min(aic_mean(aic_mean ~= 0)) - 0.25 * (max(aic_mean) - min(aic_mean(aic_mean ~= 0))), max(aic_mean) + 0.55 * (max(aic_mean) - min(aic_mean(aic_mean ~= 0)))]);

hold on

[ngroups,nbars] = size(aic_mean_reordered);

x = nan(nbars, ngroups);
for i = 1:nbars
    x(i,:) = b(i).XEndPoints;
end

er = errorbar(x', aic_mean_reordered, aic_centred_se_reordered, 'k', 'linestyle', 'none');

set(gca, 'FontSize', 9);

[leg,att] = legendflex(gca, {'Kalman filter', 'BG model'}, 'title', 'Learning rule', 'anchor', {'nw','nw'}, 'buffer', [5 -5]);

set(findall(leg, 'string', 'Learning rule'), 'fontsize', 10);

xlabel("Exploration type", "FontSize", 10);
ylabel("Mean AIC", "FontSize", 10);
h.Position(3:4) = [370, 260];
set(h,'Units','Inches');
pos = get(h,'Position');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);

% print(h,'figures/fitting_compare_aic_manuscript_fixed_only_within_subject_errorbar','-dpdf','-r0')








