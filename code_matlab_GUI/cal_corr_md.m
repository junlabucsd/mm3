function [pool_data_m, pool_data_norm_m, pool_data_d, pool_data_norm_d, pool_corr, L_corr] = cal_corr_md(input_strct)

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

%-------mother cell-------
pool_data_m = ones(length(input_strct.division_length_g2),9);
pool_data_norm_m = pool_data_m;

pool_data_m(:,1) = input_strct.septum_position_g2;
pool_data_m(:,2) = input_strct.elongation_rate_fit_g2;
pool_data_m(:,3) = input_strct.initiation_mass_g2;
pool_data_m(:,4) = input_strct.tau_cyc_g2;
pool_data_m(:,5) = input_strct.B_period_g2;
pool_data_m(:,6) = input_strct.C_period_g2;
pool_data_m(:,7) = input_strct.D_period_g2;
pool_data_m(:,8) = input_strct.division_length_g2;
pool_data_m(:,9) = input_strct.newborn_length_g2;
pool_data_m(:,10) = input_strct.generation_time_g2;
pool_data_m(:,11) = pool_data_m(:,8) - pool_data_m(:,9);

pool_data_norm_m(:,1) = pool_data_m(:,1)/mean(pool_data_m(:,1));
pool_data_norm_m(:,2) = pool_data_m(:,2)/mean(pool_data_m(:,2));
pool_data_norm_m(:,3) = pool_data_m(:,3)/mean(pool_data_m(:,3));
pool_data_norm_m(:,4) = pool_data_m(:,4)/mean(pool_data_m(:,4));
pool_data_norm_m(:,5) = pool_data_m(:,5)/mean(pool_data_m(:,5));
pool_data_norm_m(:,6) = pool_data_m(:,6)/mean(pool_data_m(:,6));
pool_data_norm_m(:,7) = pool_data_m(:,7)/mean(pool_data_m(:,7));
pool_data_norm_m(:,8) = pool_data_m(:,8)/mean(pool_data_m(:,8));
pool_data_norm_m(:,9) = pool_data_m(:,9)/mean(pool_data_m(:,9));
pool_data_norm_m(:,10) = pool_data_m(:,10)/mean(pool_data_m(:,10));
pool_data_norm_m(:,11) = pool_data_m(:,11)/mean(pool_data_m(:,11));

%-------daughter cell-------
pool_data_d = ones(length(input_strct.division_length_g3),9);
pool_data_norm_d = pool_data_d;

pool_data_d(:,1) = input_strct.septum_position_g3;
pool_data_d(:,2) = input_strct.elongation_rate_fit_g3;
pool_data_d(:,3) = input_strct.initiation_mass_g3;
pool_data_d(:,4) = input_strct.tau_cyc_g3;
pool_data_d(:,5) = input_strct.B_period_g3;
pool_data_d(:,6) = input_strct.C_period_g3;
pool_data_d(:,7) = input_strct.D_period_g3;
pool_data_d(:,8) = input_strct.division_length_g3;
pool_data_d(:,9) = input_strct.newborn_length_g3;
pool_data_d(:,10) = input_strct.generation_time_g3;
pool_data_d(:,11) = pool_data_d(:,8) - pool_data_d(:,9);

pool_data_norm_d(:,1) = pool_data_d(:,1)/mean(pool_data_d(:,1));
pool_data_norm_d(:,2) = pool_data_d(:,2)/mean(pool_data_d(:,2));
pool_data_norm_d(:,3) = pool_data_d(:,3)/mean(pool_data_d(:,3));
pool_data_norm_d(:,4) = pool_data_d(:,4)/mean(pool_data_d(:,4));
pool_data_norm_d(:,5) = pool_data_d(:,5)/mean(pool_data_d(:,5));
pool_data_norm_d(:,6) = pool_data_d(:,6)/mean(pool_data_d(:,6));
pool_data_norm_d(:,7) = pool_data_d(:,7)/mean(pool_data_d(:,7));
pool_data_norm_d(:,8) = pool_data_d(:,8)/mean(pool_data_d(:,8));
pool_data_norm_d(:,9) = pool_data_d(:,9)/mean(pool_data_d(:,9));
pool_data_norm_d(:,10) = pool_data_d(:,10)/mean(pool_data_d(:,10));
pool_data_norm_d(:,11) = pool_data_d(:,11)/mean(pool_data_d(:,11));

%% filter data
pool_data_flt_m = filter_data(pool_data_norm_m);
pool_data_flt_d = filter_data(pool_data_norm_d);

%% bin data
N_edges = 20;

pool_corr = cell(11,11);
L_corr = cell(11,11);

for i = 1:11
    [pool_corr{i,1}(:,:), L_corr{i,1}(:,:)] = bin_data(N_edges,pool_data_flt_m(:,i),pool_data_flt_d(:,i));
end

end