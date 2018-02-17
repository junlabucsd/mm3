clear all; clc;
close all;
warning off;

%% load data
cell_data = load('../../analysis/cell_data/complete_cells_foci.mat');

px_to_mu = 0.065;
t_int = 3.0;
start_cut = 0;
end_cut = 1500;

%% extract data
L = length(fieldnames(cell_data));
fnames = fieldnames(cell_data);

channel = cell(1,L);
Growth_all = cell(1,1);

N_cell = 1;
mother_cell_counter = 1;

channel_temp = cell(1,1);
Foci_all = [0 0 0];
N_foci = 0;

for i = 1:L
    
    N = length( cell_data.(fnames{i}).times );
    
    if cell_data.(fnames{i}).birth_label == 1 && ismember(cell_data.(fnames{i}).daughters(1,:),fnames) && ismember(cell_data.(fnames{i}).parent,fnames) %&& cell_data.(fnames{i}).peak > 0 && cell_data.(fnames{i}).peak < 2000 %&& cell_data.(fnames{i}).birth_time>=start_cut && cell_data.(fnames{i}).birth_time<=end_cut && N>1  %only load mother cells: cell lable r==1
       
        channel{mother_cell_counter} = cell_data.(fnames{i}); 
        
        generation_time( mother_cell_counter ) = double( channel{mother_cell_counter}.tau ) ;

        newborn_length( mother_cell_counter ) = channel{mother_cell_counter}.sb;
        newborn_width( mother_cell_counter ) = px_to_mu*channel{mother_cell_counter}.widths(1);

        % dvision length = length of dividing mother cells; dvision width = width of dividing mother cells;
        division_length( mother_cell_counter ) = channel{mother_cell_counter}.sd;
        division_width( mother_cell_counter ) = px_to_mu*channel{mother_cell_counter}.widths(end);

        newborn_volume( mother_cell_counter ) = (newborn_length( mother_cell_counter )-newborn_width( mother_cell_counter ))*pi*(newborn_width( mother_cell_counter )/2)^2+(4/3)*pi*(newborn_width( mother_cell_counter )/2)^3;
        division_volume( mother_cell_counter ) = (division_length( mother_cell_counter )-division_width( mother_cell_counter ))*pi*(division_width( mother_cell_counter )/2)^2+(4/3)*pi*(division_width( mother_cell_counter )/2)^3;

        added_length ( mother_cell_counter ) = division_length( mother_cell_counter ) - newborn_length( mother_cell_counter );

        added_volume ( mother_cell_counter ) = division_volume( mother_cell_counter ) - newborn_volume( mother_cell_counter );

        cell_width( mother_cell_counter ) = px_to_mu*mean(channel{mother_cell_counter}.widths(1:end));

        septum_position ( mother_cell_counter ) = channel{mother_cell_counter}.septum_position;

        birth_time( mother_cell_counter) = t_int*double(channel{mother_cell_counter}.birth_time);
        division_time( mother_cell_counter) = t_int*double(channel{mother_cell_counter}.division_time);

        growth_rate( mother_cell_counter ) = 60*log(division_volume( mother_cell_counter )/newborn_volume( mother_cell_counter ))/generation_time( mother_cell_counter );
        elongation_rate( mother_cell_counter ) = 60*log(division_length( mother_cell_counter )/newborn_length( mother_cell_counter ))/generation_time( mother_cell_counter );
        
        if N>2
            ft1 = fittype('a*x+b');
            Growth_time = t_int*double( channel{mother_cell_counter}.times_w_div - channel{mother_cell_counter}.times_w_div(1) );
            Growth_length = px_to_mu*channel{mother_cell_counter}.lengths_w_div; 

            fit_temp = fit(Growth_time',log(Growth_length)',ft1);

            elongation_rate_fit( mother_cell_counter ) = 60*fit_temp.a;
        elseif N==2
            elongation_rate_fit( mother_cell_counter ) = 60*elongation_rate( mother_cell_counter );
        end

        mother_cell_counter = mother_cell_counter + 1;      
        
    end
    
    if mod(i,100)==0
        i
    end
    
end

save('../../analysis/cal_volume_mm3.mat');
