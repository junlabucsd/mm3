% Export the array output data from FS stat_cell_param.m to .csv

% directory from which to load and save data
% data_dir = '../../analysis/cell_data/';
% data_dir = '/Volumes/JunLabSSD_04/bsub_unitcell/20180727_BS43_1/analysis/cell_data/';
% data_dir = '/Volumes/JunLabSSD_04/bsub_unitcell/20180728_BS43_2/analysis/cell_data/';
% data_dir = '/Volumes/JunLabSSD_04/bsub_unitcell/20180803_BS43_3/analysis/cell_data/';
data_dir = '/Volumes/JunLabSSD_04/bsub_unitcell/20180809_BS43_4/analysis/cell_data/';

% load output from stat_cell_param
% load([data_dir 'cell_cycle_stat_GUI.mat']);

% table header
header = {'cell_id', 'initiation_time', 'initiation_length', ...
          'termination_time', 'B', 'C', 'D', 'tau_cyc', 'n_oc', 'width', ...
          'tau', 'birth_time', 'division_time', 'elong_rate', ...
          'birth_length', 'division_length', 'delta', 'septum_position'};

% make table
T = table(cell_id', initiation_time_m', initiation_mass_m', ...
          termination_time', B_period', C_period', D_period', tau_cyc', ...
          n_oc', cell_width', generation_time', birth_time', division_time', elongation_rate_fit', ...
          newborn_length', division_length', added_length', septum_position');

% add header
T.Properties.VariableNames = header;

% save to csv
writetable(T, [data_dir 'cell_cycle_data.csv'], 'Delimiter', ',');
