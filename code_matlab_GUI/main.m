clear all; clc;
% close all;
warning off;

%% test mm3 
% % % % load data
% file_dir = '/Users/Fangwei/Documents/Adder/Analysis/20170623_FS103_DnaN_YPet_glu_arg_25p_200ms_SJW103/analysis/';
% EXP0 = load([file_dir 'cell_cycle_stat_GUI.mat']);
% file_name = [file_dir '20170623_cell_cycle_stat_GUI.mat'];
% %         
% % Extract data
% [EXP0_data, EXP0_data_norm, EXP0_CV, EXP0_corr, EXP0_L] = cal_corr(EXP0);
% 
% save(file_name);
% 
% % plot
% load(file_name);
% 
% plot_corr(EXP0_corr, EXP0_data_norm);
% % text(-4.5, 12, 'DnaN-yPet','FontSize',30);
% 
% plot_dist(EXP0_data, EXP0_data_norm, EXP0_CV);
% % text(-4.5, 12, 'DnaN-yPet','FontSize',30);
% 
% % save([file_dir 'mm3_20170623_manual.txt'],'EXP0_data','-ascii');

%% test mm3 - mother-daughter
% % load data
file_dir = '/Users/Fangwei/Documents/Adder/Analysis/20170623_FS103_DnaN_YPet_glu_arg_25p_200ms_SJW103/analysis/';
EXP0 = load([file_dir 'cell_cycle_stat_md_GUI_all.mat']);
file_name = [file_dir '20170623_cell_cycle_stat_GUI_all.mat'];
        
% Extract data
[EXP0_data_g2, EXP0_data_norm_g2, EXP0_CV_g2, EXP0_corr_g2, EXP0_L_g2] = cal_corr_g2(EXP0);
[EXP0_data_g3, EXP0_data_norm_g3, EXP0_CV_g3, EXP0_corr_g3, EXP0_L_g3] = cal_corr_g3(EXP0);
% 
[EXP0_data_m, EXP0_data_norm_m, EXP0_data_d, EXP0_data_norm_d, EXP0_pool_corr_md, EXP0_L_corr_md] = cal_corr_md(EXP0);
% 
% save(file_name);
% 
% % plot
% load(file_name);

plot_corr_g2(EXP0_corr_g2, EXP0_data_norm_g2);
text(-6.5, 15.5, 'mother cells','FontSize',30);

plot_dist_g2(EXP0_data_g2, EXP0_data_norm_g2, EXP0_CV_g2);
text(-6.5, 0.6, 'mother cells','FontSize',30);

plot_corr_g2(EXP0_corr_g3, EXP0_data_norm_g3);
text(-6.5, 15.5, 'daughter cells','FontSize',30);

plot_dist_g2(EXP0_data_g3, EXP0_data_norm_g3, EXP0_CV_g3);
text(-6.5, 0.6, 'daughter cells','FontSize',30);

plot_corr_md(EXP0_pool_corr_md, EXP0_data_norm_m, EXP0_data_norm_d)
text(-6.5, 1.9, 'mother-daughter correlation','FontSize',30);

save([file_dir 'mm3_20170623_g1_manual.txt'],'EXP0_data_g2','-ascii');
save([file_dir 'mm3_20170623_g2_manual.txt'],'EXP0_data_g3','-ascii');