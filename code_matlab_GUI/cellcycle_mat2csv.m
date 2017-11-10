% Export the array output data from FS stat_cell_param.m to .csv

% directory from which to load and save data
data_dir = '/Volumes/JunLabSSD_04/shift/ecoli/20171020_ecoli_shift14/analysis/cell_data/';

% load output from stat_cell_param
% load([data_dir 'cell_cycle_stat_GUI.mat']);

% table header
header = {'cell_id', 'initiation_time', 'initiation_length', ...
          'termination_time', 'B', 'C', 'D', 'tau_cyc', 'width'}; 

% make table
T = table(cell_id', initiation_time_m', initiation_mass_m', ...
          termination_time', B_period', C_period', D_period', tau_cyc', cell_width'); 
      
% add header
T.Properties.VariableNames = header;

% save to csv
writetable(T, [data_dir 'cell_cycle_data.csv'], 'Delimiter', ',');