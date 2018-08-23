clear all; clc;
close all;
warning off;

%% load data

dir_name = '../../analysis/picked/';;
fnames = dir( [ dir_name '/*.mat' ]);

px_to_mu = 0.065;
t_int = 3.0;

%% extract and calculate all cell data

mother_cell_counter = 1;

for i=1:numel(fnames)
    struct_tmp = load([dir_name fnames(i).name]);
    fnames_channel = fieldnames(struct_tmp.cell_list);
    L_channles = length(fnames_channel(:,1));

    for j=1:L_channles
        cell_g2_id = fnames_channel{j,1};
        cell_g2_temp = struct_tmp.cell_list.(cell_g2_id);

        length_g2_temp = double(px_to_mu*cell_g2_temp.lengths_w_div);

%         if cell_g2_temp.birth_label ==1 && isfield(cell_g2_temp,'initiation_time') == 1 && cell_g2_temp.lengths_w_div(end) < 20 && ismember(cell_g2_temp.daughters(1,:),fnames_channel) %&& cell_g2_temp.peak > 0 && cell_g2_temp.peak < 2000  %filter out filamentous cells
        if cell_g2_temp.birth_label ==1 && isfield(cell_g2_temp,'initiation_time') == 1 && isfield(cell_g2_temp,'initiation_time_n') == 1 && cell_g2_temp.lengths_w_div(end) < 20 && ismember(cell_g2_temp.daughters(1,:),fnames_channel) %&& cell_g2_temp.peak > 0 && cell_g2_temp.peak < 2000  %filter out filamentous cells
%         if cell_g2_temp.birth_label ==1 && isfield(cell_g2_temp,'initiation_time') == 1 && isfield(cell_g2_temp,'initiation_time_n') == 1 && isfield(cell_g2_temp,'initiation_time_n2') == 0 && cell_g2_temp.n_oc_n == 2 && cell_g2_temp.lengths_w_div(end) < 20 && ismember(cell_g2_temp.daughters(1,:),fnames_channel) %&& cell_g2_temp.peak > 0 && cell_g2_temp.peak < 2000  %filter out filamentous cells


            cell_g3_id = cell_g2_temp.daughters(1,:);
            cell_g3_temp = struct_tmp.cell_list.(cell_g3_id);

            length_g3_temp = double(px_to_mu*cell_g3_temp.lengths_w_div);

%             if cell_g2_temp.lengths_w_div(end) < 20 && isfield(cell_g3_temp,'initiation_time') == 1 && cell_g3_temp.lengths_w_div(end) < 20 %filter out filamentous cells
            if cell_g2_temp.lengths_w_div(end) < 20 && isfield(cell_g3_temp,'initiation_time') == 1  && isfield(cell_g3_temp,'initiation_time_n') == 1 && cell_g3_temp.lengths_w_div(end) < 20 %filter out filamentous cells
%             if cell_g2_temp.lengths_w_div(end) < 20 && isfield(cell_g3_temp,'initiation_time') == 1  && isfield(cell_g3_temp,'initiation_time_n') == 1 && isfield(cell_g3_temp,'initiation_time_n2') == 0 && cell_g3_temp.n_oc_n == 2 && cell_g3_temp.lengths_w_div(end) < 20 %filter out filamentous cells



                %----------parameters for mother generation--------
                generation_time_g2( mother_cell_counter ) = double( cell_g2_temp.tau ) ;

                newborn_length_g2( mother_cell_counter ) = cell_g2_temp.sb;
                newborn_width_g2( mother_cell_counter ) = cell_g2_temp.widths_w_div(1);

                division_length_g2( mother_cell_counter ) = cell_g2_temp.sd;
                division_width_g2( mother_cell_counter ) = cell_g2_temp.widths_w_div(end);

                newborn_volume_g2( mother_cell_counter ) = (newborn_length_g2( mother_cell_counter )-newborn_width_g2( mother_cell_counter ))*pi*(newborn_width_g2( mother_cell_counter )/2)^2+(4/3)*pi*(newborn_width_g2( mother_cell_counter )/2)^3;
                division_volume_g2( mother_cell_counter ) = (division_length_g2( mother_cell_counter )-division_width_g2( mother_cell_counter ))*pi*(division_width_g2( mother_cell_counter )/2)^2+(4/3)*pi*(division_width_g2( mother_cell_counter )/2)^3;

                added_length_g2( mother_cell_counter ) = division_length_g2( mother_cell_counter ) - newborn_length_g2( mother_cell_counter );

                added_volume_g2( mother_cell_counter ) = division_volume_g2( mother_cell_counter ) - newborn_volume_g2( mother_cell_counter );

                cell_width_g2( mother_cell_counter ) = mean(cell_g2_temp.widths_w_div(1:end));

                septum_position_g2( mother_cell_counter ) = cell_g2_temp.septum_position;

                birth_time_g2( mother_cell_counter) = t_int*double(cell_g2_temp.birth_time);
                division_time_g2( mother_cell_counter) = t_int*double(cell_g2_temp.division_time);

                growth_rate_g2( mother_cell_counter ) = 60*log(division_volume_g2( mother_cell_counter )/newborn_volume_g2( mother_cell_counter ))/generation_time_g2( mother_cell_counter );
                elongation_rate_g2( mother_cell_counter ) = 60*log(division_length_g2( mother_cell_counter )/newborn_length_g2( mother_cell_counter ))/generation_time_g2( mother_cell_counter );

                if length(length_g2_temp)>2
                    ft1 = fittype('a*x+b');
                    Growth_time = t_int*double( cell_g2_temp.times_w_div - cell_g2_temp.times_w_div(1) );
                    Growth_length = px_to_mu*cell_g2_temp.lengths_w_div;

                    fit_temp = fit(Growth_time',log(Growth_length)',ft1);

                    elongation_rate_fit_g2( mother_cell_counter ) = 60*fit_temp.a;
                elseif length(length_g2_temp)==2
                    elongation_rate_fit_g2( mother_cell_counter ) = 60*elongation_rate_g2( mother_cell_counter );
                end

                initiation_time_g2_m( mother_cell_counter ) = t_int*double(cell_g2_temp.initiation_time); %note the change in definitions
                initiation_time_g2( mother_cell_counter ) = t_int*double(cell_g2_temp.initiation_time_n);
                termination_time_g2( mother_cell_counter ) = t_int*double(cell_g2_temp.termination_time);

                initiation_mass_g2_m( mother_cell_counter ) = cell_g2_temp.initiation_mass; %note the change in definitions
                initiation_mass_g2( mother_cell_counter ) = cell_g2_temp.initiation_mass_n;
                termination_mass_g2( mother_cell_counter ) = cell_g2_temp.termination_mass;

                B_period_g2( mother_cell_counter ) = t_int*double(cell_g2_temp.initiation_time - cell_g2_temp.birth_time_m);
                C_period_g2( mother_cell_counter ) = t_int*double(cell_g2_temp.termination_time - cell_g2_temp.initiation_time);
                D_period_g2( mother_cell_counter ) = t_int*double(cell_g2_temp.division_time - cell_g2_temp.termination_time);
                tau_cyc_g2( mother_cell_counter ) = t_int*double(cell_g2_temp.division_time - cell_g2_temp.initiation_time);


                %----------parameters for daughter generation--------
                generation_time_g3( mother_cell_counter ) = double( cell_g3_temp.tau ) ;

                newborn_length_g3( mother_cell_counter ) = cell_g3_temp.sb;
                newborn_width_g3( mother_cell_counter ) = cell_g3_temp.widths_w_div(1);

                division_length_g3( mother_cell_counter ) = cell_g3_temp.sd;
                division_width_g3( mother_cell_counter ) = cell_g3_temp.widths_w_div(end);

                newborn_volume_g3( mother_cell_counter ) = (newborn_length_g3( mother_cell_counter )-newborn_width_g3( mother_cell_counter ))*pi*(newborn_width_g3( mother_cell_counter )/2)^2+(4/3)*pi*(newborn_width_g3( mother_cell_counter )/2)^3;
                division_volume_g3( mother_cell_counter ) = (division_length_g3( mother_cell_counter )-division_width_g3( mother_cell_counter ))*pi*(division_width_g3( mother_cell_counter )/2)^2+(4/3)*pi*(division_width_g3( mother_cell_counter )/2)^3;

                added_length_g3( mother_cell_counter ) = division_length_g3( mother_cell_counter ) - newborn_length_g3( mother_cell_counter );

                added_volume_g3( mother_cell_counter ) = division_volume_g3( mother_cell_counter ) - newborn_volume_g3( mother_cell_counter );

                cell_width_g3( mother_cell_counter ) = mean(cell_g3_temp.widths_w_div(1:end));

                septum_position_g3( mother_cell_counter ) = cell_g3_temp.septum_position;

                birth_time_g3( mother_cell_counter) = t_int*double(cell_g3_temp.birth_time);
                division_time_g3( mother_cell_counter) = t_int*double(cell_g3_temp.division_time);

                growth_rate_g3( mother_cell_counter ) = 60*log(division_volume_g3( mother_cell_counter )/newborn_volume_g3( mother_cell_counter ))/generation_time_g3( mother_cell_counter );
                elongation_rate_g3( mother_cell_counter ) = 60*log(division_length_g3( mother_cell_counter )/newborn_length_g3( mother_cell_counter ))/generation_time_g3( mother_cell_counter );

                if length(length_g3_temp)>2
                    ft1 = fittype('a*x+b');
                    Growth_time = t_int*double( cell_g3_temp.times_w_div - cell_g3_temp.times_w_div(1) );
                    Growth_length = px_to_mu*cell_g3_temp.lengths_w_div;

                    fit_temp = fit(Growth_time',log(Growth_length)',ft1);

                    elongation_rate_fit_g3( mother_cell_counter ) = 60*fit_temp.a;
                elseif length(length_g3_temp)==2
                    elongation_rate_fit_g3( mother_cell_counter ) = 60*elongation_rate_g3( mother_cell_counter );
                end

                initiation_time_g3_m( mother_cell_counter ) = t_int*double(cell_g3_temp.initiation_time); %note the change in definitions
                initiation_time_g3( mother_cell_counter ) = t_int*double(cell_g3_temp.initiation_time_n);
                termination_time_g3( mother_cell_counter ) = t_int*double(cell_g3_temp.termination_time);

                initiation_mass_g3_m( mother_cell_counter ) = cell_g3_temp.initiation_mass; %note the change in definitions
                initiation_mass_g3( mother_cell_counter ) = cell_g3_temp.initiation_mass_n;
                termination_mass_g3( mother_cell_counter ) = cell_g3_temp.termination_mass;

                B_period_g3( mother_cell_counter ) = t_int*double(cell_g3_temp.initiation_time - cell_g3_temp.birth_time_m);
                C_period_g3( mother_cell_counter ) = t_int*double(cell_g3_temp.termination_time - cell_g3_temp.initiation_time);
                D_period_g3( mother_cell_counter ) = t_int*double(cell_g3_temp.division_time - cell_g3_temp.termination_time);
                tau_cyc_g3( mother_cell_counter ) = t_int*double(cell_g3_temp.division_time - cell_g3_temp.initiation_time);

                mother_cell_counter = mother_cell_counter + 1;

            end


        end

    end
    if mod(i,10)==0
        i
    end
end

% save('../../analysis/cell_cycle_stat_md_GUI_noc2.mat');
