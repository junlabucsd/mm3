function [pool_data, pool_data_norm, pool_CV, pool_corr, L_corr] = cal_corr_g3(input_strct)

%% Extract parameters and calculate CVs
% 1 - septum position
% 2 - elongation rate
% 3 - initiation mass
% 4 - cell cycle duration
% 5 - B period
% 6 - C period
% 7 - D period
% 8 - division volume
% 9 - newborn volume
% 10 - generation time
% 11 - added volume
% 12 - added volume between initiations


pool_data = ones(length(input_strct.division_length_g3),9);
pool_data_norm = pool_data;
pool_CV = zeros(1,12);
pool_data(:,1) = input_strct.septum_position_g3;
pool_data(:,2) = input_strct.elongation_rate_fit_g3;
pool_data(:,3) = input_strct.initiation_mass_g3;
pool_data(:,4) = input_strct.tau_cyc_g3;
pool_data(:,5) = input_strct.B_period_g3;
pool_data(:,6) = input_strct.C_period_g3;
pool_data(:,7) = input_strct.D_period_g3;
pool_data(:,8) = input_strct.division_length_g3;
pool_data(:,9) = input_strct.newborn_length_g3;
% pool_data(:,9) = 0.5*input_strct.newborn_length_g2;
pool_data(:,10) = input_strct.generation_time_g3;
pool_data(:,11) = pool_data(:,8) - pool_data(:,9);
pool_data(:,12) = 2*input_strct.initiation_mass_g3_m - input_strct.initiation_mass_g2_m;

pool_data_norm(:,1) = pool_data(:,1)/nanmean(pool_data(:,1));
pool_data_norm(:,2) = pool_data(:,2)/nanmean(pool_data(:,2));
pool_data_norm(:,3) = pool_data(:,3)/nanmean(pool_data(:,3));
pool_data_norm(:,4) = pool_data(:,4)/nanmean(pool_data(:,4));
pool_data_norm(:,5) = pool_data(:,5)/nanmean(pool_data(:,5));
pool_data_norm(:,6) = pool_data(:,6)/nanmean(pool_data(:,6));
pool_data_norm(:,7) = pool_data(:,7)/nanmean(pool_data(:,7));
pool_data_norm(:,8) = pool_data(:,8)/nanmean(pool_data(:,8));
pool_data_norm(:,9) = pool_data(:,9)/nanmean(pool_data(:,9));
pool_data_norm(:,10) = pool_data(:,10)/nanmean(pool_data(:,10));
pool_data_norm(:,11) = pool_data(:,11)/nanmean(pool_data(:,11));
pool_data_norm(:,12) = pool_data(:,12)/nanmean(pool_data(:,12));

pool_CV(:,1) = nanstd(pool_data_norm(:,1));
pool_CV(:,2) = nanstd(pool_data_norm(:,2));
pool_CV(:,3) = nanstd(pool_data_norm(:,3));
pool_CV(:,4) = nanstd(pool_data_norm(:,4));
pool_CV(:,5) = nanstd(pool_data_norm(:,5));
pool_CV(:,6) = nanstd(pool_data_norm(:,6));
pool_CV(:,7) = nanstd(pool_data_norm(:,7));
pool_CV(:,8) = nanstd(pool_data_norm(:,8));
pool_CV(:,9) = nanstd(pool_data_norm(:,9));
pool_CV(:,10) = nanstd(pool_data_norm(:,10));
pool_CV(:,11) = nanstd(pool_data_norm(:,11));
pool_CV(:,12) = nanstd(pool_data_norm(:,12));

%% filter data
pool_data_flt = filter_data(pool_data_norm);

%% bin data
N_edges = 20;

pool_corr = cell(12,12);
L_corr = cell(12,12);

k = 1;
for i = 1:12
    for j = 1:12
        if i~=j
            [pool_corr{i,j}(:,:), L_corr{i,j}(:,:)] = bin_data(N_edges,pool_data_flt(:,i),pool_data_flt(:,j));
        end
%         k = k+1
    end
end

end