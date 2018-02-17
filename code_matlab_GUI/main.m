clear all; clc;
close all;
warning off;

%% test mm3 
% %Note: all cells seems at steady-state from beginning to the end 
% 
% % % % load data
% file_dir = '../../analysis/';
% EXP0 = load([file_dir 'cell_cycle_stat_GUI.mat']);
% file_name = [file_dir '20171026_cell_cycle_stat_GUI.mat'];
% %         
% % Extract data
% [EXP0_data, EXP0_data_norm, EXP0_CV, EXP0_corr, EXP0_L] = cal_corr(EXP0);
% % % 
% % save(file_name);
% % 
% % % plot
% % load(file_name);
% 
% plot_corr(EXP0_corr, EXP0_data_norm);
% % text(-4.5, 12, 'DnaN-yPet','FontSize',30);
% 
% plot_dist(EXP0_data, EXP0_data_norm, EXP0_CV);
% % text(-4.5, 12, 'DnaN-yPet','FontSize',30);
% 
% % save([file_dir 'mm3_20171026_manual.txt'],'EXP0_data','-ascii');

%% test mm3 - mother-daughter
% % load data
file_dir = '../../analysis/';
EXP0 = load([file_dir 'cell_cycle_stat_md_GUI_text.mat']);
% file_name = [file_dir '20171026_cell_cycle_stat_GUI.mat'];
        
% Extract data
[EXP0_data_g2, EXP0_data_norm_g2, EXP0_CV_g2, EXP0_corr_g2, EXP0_L_g2] = cal_corr_g2(EXP0);
[EXP0_data_g3, EXP0_data_norm_g3, EXP0_CV_g3, EXP0_corr_g3, EXP0_L_g3] = cal_corr_g3(EXP0);

[EXP0_data_m, EXP0_data_norm_m, EXP0_data_d, EXP0_data_norm_d, EXP0_pool_corr_md, EXP0_L_corr_md] = cal_corr_md(EXP0);
% 
% save(file_name);
% 
% % plot
% load(file_name);

plot_corr_g2(EXP0_corr_g2, EXP0_data_norm_g2);
text(-6.5, 15.5, 'mother cells','FontSize',30);

% figHandles = findobj('Type', 'figure');
% hgexport(figHandles(3),'../../results/corr_scattered_m.png',hgexport('factorystyle'), 'Format', 'png'); 
% hgexport(figHandles(2),'../../results/corr_hist2_m.png',hgexport('factorystyle'), 'Format', 'png'); 
% hgexport(figHandles(1),'../../results/corr_binned_m.png',hgexport('factorystyle'), 'Format', 'png'); 
% close all;

plot_dist_g2(EXP0_data_g2, EXP0_data_norm_g2, EXP0_CV_g2);
text(-6.5, 0.6, 'mother cells','FontSize',30);

% figHandles = findobj('Type', 'figure');
% hgexport(figHandles(2),'../../results/dist_raw_m.png',hgexport('factorystyle'), 'Format', 'png'); 
% hgexport(figHandles(1),'../../results/dist_norm_m.png',hgexport('factorystyle'), 'Format', 'png'); 
% close all;

plot_corr_g2(EXP0_corr_g3, EXP0_data_norm_g3);
text(-6.5, 15.5, 'daughter cells','FontSize',30);

% figHandles = findobj('Type', 'figure');
% hgexport(figHandles(3),'../../results/corr_scattered_d.png',hgexport('factorystyle'), 'Format', 'png'); 
% hgexport(figHandles(2),'../../results/corr_hist2_d.png',hgexport('factorystyle'), 'Format', 'png'); 
% hgexport(figHandles(1),'../../results/corr_binned_d.png',hgexport('factorystyle'), 'Format', 'png'); 
% close all;

plot_dist_g2(EXP0_data_g3, EXP0_data_norm_g3, EXP0_CV_g3);
text(-6.5, 0.6, 'daughter cells','FontSize',30);

% figHandles = findobj('Type', 'figure');
% hgexport(figHandles(2),'../../results/dist_raw_d.png',hgexport('factorystyle'), 'Format', 'png'); 
% hgexport(figHandles(1),'../../results/dist_norm_d.png',hgexport('factorystyle'), 'Format', 'png'); 
% close all;

plot_corr_md(EXP0_pool_corr_md, EXP0_data_norm_m, EXP0_data_norm_d)
text(-8.5, 15.5, 'mother-daughter correlation','FontSize',30);

% figHandles = findobj('Type', 'figure');
% hgexport(figHandles(3),'../../results/corr_scattered_md.png',hgexport('factorystyle'), 'Format', 'png'); 
% hgexport(figHandles(2),'../../results/corr_hist2_md.png',hgexport('factorystyle'), 'Format', 'png'); 
% hgexport(figHandles(1),'../../results/corr_binned_md.png',hgexport('factorystyle'), 'Format', 'png'); 
% close all;

plot_corr_auto(EXP0_pool_corr_md, EXP0_data_norm_m, EXP0_data_norm_d)
text(-6.5, 1.9, 'autocorrelation','FontSize',30);

% figHandles = findobj('Type', 'figure');
% hgexport(figHandles(3),'../../results/corr_scattered_auto.png',hgexport('factorystyle'), 'Format', 'png'); 
% hgexport(figHandles(2),'../../results/corr_hist2_auto.png',hgexport('factorystyle'), 'Format', 'png'); 
% hgexport(figHandles(1),'../../results/corr_binned_auto.png',hgexport('factorystyle'), 'Format', 'png'); 
% close all;

% save([file_dir 'mm3_20171026_g1_manual.txt'],'EXP0_data_g2','-ascii');
% save([file_dir 'mm3_20171026_g2_manual.txt'],'EXP0_data_g3','-ascii');