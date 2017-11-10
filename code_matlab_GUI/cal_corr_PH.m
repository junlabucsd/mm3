function [pool_data, pool_data_norm, pool_CV, pool_corr, L_corr] = cal_corr_PH(input_strct)

%% Extract parameters and calculate CVs
% 1 - septum position
% 2 - elongation rate
% 3 - initiation mass
% 4 - cell cycle duration
% 5 - division volume
% 6 - newborn volume
% 7 - generation time
% 8 - added volume
% 9 - added volume between initiations (TBA)

pool_data = ones(length(input_strct.division_length),9);
pool_data_norm = pool_data;
pool_CV = zeros(1,9);

pool_data(:,1) = input_strct.septum_position;
pool_data(:,2) = input_strct.elongation_rate;
% pool_data(:,3) = input_strct.initiation_mass;
% pool_data(:,4) = input_strct.tau_cycle;
pool_data(:,5) = input_strct.division_length;
pool_data(:,6) = input_strct.newborn_length;
pool_data(:,7) = input_strct.generation_time;
pool_data(:,8) = pool_data(:,5) - pool_data(:,6);
% pool_data(:,9) = input_strct.initiation_time;

pool_data_norm(:,1) = pool_data(:,1)/mean(pool_data(:,1));
pool_data_norm(:,2) = pool_data(:,2)/mean(pool_data(:,2));
pool_data_norm(:,3) = pool_data(:,3)/mean(pool_data(:,3));
pool_data_norm(:,4) = pool_data(:,4)/mean(pool_data(:,4));
pool_data_norm(:,5) = pool_data(:,5)/mean(pool_data(:,5));
pool_data_norm(:,6) = pool_data(:,6)/mean(pool_data(:,6));
pool_data_norm(:,7) = pool_data(:,7)/mean(pool_data(:,7));
pool_data_norm(:,8) = pool_data(:,8)/mean(pool_data(:,8));
pool_data_norm(:,9) = pool_data(:,9)/mean(pool_data(:,9));

pool_CV(:,1) = std(pool_data_norm(:,1));
pool_CV(:,2) = std(pool_data_norm(:,2));
pool_CV(:,3) = std(pool_data_norm(:,3));
pool_CV(:,4) = std(pool_data_norm(:,4));
pool_CV(:,5) = std(pool_data_norm(:,5));
pool_CV(:,6) = std(pool_data_norm(:,6));
pool_CV(:,7) = std(pool_data_norm(:,7));
pool_CV(:,8) = std(pool_data_norm(:,8));
pool_CV(:,9) = std(pool_data_norm(:,9));

%% filter data
pool_data_flt = filter_data(pool_data_norm);

%% bin data
N_edges = 20;

pool_corr = cell(9,9);
L_corr = cell(9,9);

k = 1;
for i = 1:9
    for j = 1:9
        if i~=j
            [pool_corr{i,j}(:,:), L_corr{i,j}(:,:)] = bin_data(N_edges,pool_data_flt(:,i),pool_data_flt(:,j));
        end
%         k = k+1
    end
end

end