clear all; clc;
close all;
warning off;

%% test mm3 
% % % load data
file_dir = '../../analysis/';
EXP0 = load([file_dir 'cal_volume_mm3.mat']);
file_name = [file_dir '20171026_EXP0_DnaN.mat'];
file_name_pc = [file_dir '20171026_EXP0_DnaN_pc.mat'];
file_name_fl = [file_dir '20171026_EXP0_DnaN_fl.mat'];
%         
% Extract data
[EXP0_data, EXP0_data_norm, EXP0_CV, EXP0_corr, EXP0_L] = cal_corr_PH(EXP0);
% 
% save(file_name);

% % plot
% load(file_name);

plot_corr_PH(EXP0_corr, EXP0_data_norm);
% text(-4.5, 12, 'mm3','FontSize',30);
% 
plot_dist_PH(EXP0_data, EXP0_data_norm,EXP0_CV);
% text(-4.5, 12, 'DnaN-yPet','FontSize',30);
% 
plot_time(EXP0)

% save([file_dir 'mm3.txt'],'EXP0_data','-ascii');