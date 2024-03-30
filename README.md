This repository supports the article "Dopamine encoding of novelty facilitates efficient uncertainty-driven exploration".

The repository consists of three sections, each contained in one of the sub-folders in the root directory. The first contains neural data (in vivo electrophysiological recording) from two non-human primate subjects, in MATLAB .mat format, with analysis code in .m format. The second contains behavioral data from two experiments with 45 and 44 human participants, in .mat format with analysis code in .m format. The third are model simulations, with a Jupyter Notebook file (.ipynb) to execute simulations, and  function definitions in .py format. 

## Neural data analysis

This folder corresponds to the analysis of electrophysiological recording data first published in [Lak et al. (2016)](http://dx.doi.org/10.7554/eLife.18044). The raw data is also [available](https://www.laklab.org/open-science.html) from the original authors. 

firing_rates_conditioning.mat: firing rate data from conditioning task, as published with [Lak et al. (2016)](http://dx.doi.org/10.7554/eLife.18044). Consists of normalised firing rates of individual neurons recorded in different sessions. 
firing_rates_choice_task.mat: firing rate data recovered from Figure 6 in [Lak et al. (2016)](http://dx.doi.org/10.7554/eLife.18044), which represents average firing rates over multiple neurons.
conditioning_results.mat: results of function fitting using the conditioning data.
choice_results.mat: results of function fitting using the recovered choice task data.
fitting_conditioning.m: analysis script used to generate conditioning_results.mat.
fitting_choice_task.m: analysis script used to generate choice_results.mat.
plotting_conditioning.m: script to generate plots using results in conditioning_results.mat.
plotting_choice.m: script to generate plots using results in choice_results.mat.

All other files are non-executables defining functions.

Tested with MATLAB R2023b.
## Behavior analysis

This folder corresponds to the analysis of behavior data first published in [Gershman (2018)](https://doi.org/10.1016/j.cognition.2017.12.014). This folder includes partially adapted code. The raw data and code used for analysis in [Gershman (2018)](https://doi.org/10.1016/j.cognition.2017.12.014) are also [available](https://github.com/sjgershm/exploration) from the original author. The analysis in [Gershman (2018)](https://doi.org/10.1016/j.cognition.2017.12.014) uses a fitting [toolbox](https://github.com/sjgershm/mfit), which is included in the folder.

data1.csv, data2.csv: raw data from two behavioral experiments with human participants. Both experiments involve two-armed bandit tasks of different set-ups. Only data2.csv is used for analysis in this work.
main_scripts.m: scripts used to run model fitting analysis and generate plots with the results. 
all_results.mat: all numerical results from model fitting analysis.
results_sim1.mat, results_sim2.mat: analysis results from [Gershman (2018)](https://doi.org/10.1016/j.cognition.2017.12.014).

All other files are function definitions. 

Tested with MATLAB R2023b.
## Model simulations

This folder corresponds to the performance comparison of different exploration algorithms in simulations.

simulation.ipynb: contains code to run simulations using optimal parameters and generate plots.
simulations.py: function definitions.
results: folder containing the numerical simulation results. 

All other files are scripts used to find optimal parameters of various models for each individual task.

Tested with Python 3.10.