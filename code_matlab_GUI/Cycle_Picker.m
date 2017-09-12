function varargout = Cycle_Picker(varargin)
% CYCLE_PICKER MATLAB code for Cycle_Picker.fig
%      CYCLE_PICKER, by itself, creates a new CYCLE_PICKER or raises the existing
%      singleton*.
%
%      H = CYCLE_PICKER returns the handle to a new CYCLE_PICKER or the handle to
%      the existing singleton*.
%
%      CYCLE_PICKER('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in CYCLE_PICKER.M with the given input arguments.
%
%      CYCLE_PICKER('Property','Value',...) creates a new CYCLE_PICKER or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before Cycle_Picker_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to Cycle_Picker_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help Cycle_Picker

% Last Modified by GUIDE v2.5 31-Aug-2017 15:12:29

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @Cycle_Picker_OpeningFcn, ...
                   'gui_OutputFcn',  @Cycle_Picker_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before Cycle_Picker is made visible.
function Cycle_Picker_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to Cycle_Picker (see VARARGIN)

% clear all;
clc;
% close all;

%-------------start pre-prosessing-----------
handles.dir_name = '../../analysis/';
handles.cell_data = load([handles.dir_name 'cell_data/all_cells_foci.mat']);
handles.px_to_mu = 0.065;
handles.IW_thr = 6587; % threshold of intensity weighting
handles.n_oc = 3; %number of overlapping cell cycle

handles.xlim_max = 200;
handles.ylim_max = 6;
handles.time_int = 2;

if exist([handles.dir_name 'picked/']) == 0
    mkdir([handles.dir_name 'picked/']);
end

if exist([handles.dir_name 'picked_png/']) == 0
    mkdir([handles.dir_name 'picked_png/']);
end



L = length(fieldnames(handles.cell_data));
fnames = fieldnames(handles.cell_data);

fnames_array = cell2mat(fnames);

fnames_array_num = zeros(L,4);

fnames_array_num(:,1) = str2num(fnames_array(:,2:3));
fnames_array_num(:,2) = str2num(fnames_array(:,5:8));
fnames_array_num(:,3) = str2num(fnames_array(:,10:13));
fnames_array_num(:,4) = str2num(fnames_array(:,15:16));

fnames_sort = sortrows(fnames_array_num,4);
fnames_sort = sortrows(fnames_sort,3);
fnames_sort = sortrows(fnames_sort,2);
fnames_sort = sortrows(fnames_sort,1);

fovs = unique(fnames_sort(:,1));

handles.channels = [0 0];
for i = fovs'
    fov_index = find(fnames_sort(:,1) == i);
    fnames_fov = fnames_sort(fov_index,:);
    
    channels_tmp(:,2) = unique(fnames_fov(:,2));
    channels_tmp(:,1) = i + zeros(length(channels_tmp(:,2)),1);
    
    handles.channels = [handles.channels; channels_tmp];
    clear channels_tmp;
end
handles.channels(1,:) = [];

handles.fnames_sort = fnames_sort;
handles.channle_idx = 1;

[handles.length_list, handles.foci_list, handles.birth_list, handles.division_list, handles.cell_list, handles.cell_names, handles.save_name, handles.save_name_png, handles.display_name] ...
    = plot_channel(handles.dir_name, handles.cell_data, handles.px_to_mu, handles.IW_thr, handles.fnames_sort, handles.channels, handles.channle_idx, handles.xlim_max, handles.ylim_max, handles.time_int);

set(handles.display, 'String' , handles.display_name );

handles.clicks = 0;
handles.initiation_time = [];
handles.termination_time = [];
handles.division_time = [];

handles.birth_time_m = [];
handles.initiation_mass = [];
handles.termination_mass = [];

%-------------end pre-prosessing-----------

% Choose default command line output for Cycle_Picker
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes Cycle_Picker wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = Cycle_Picker_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes during object creation, after setting all properties.
function axes2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to axes2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
hold on;
% Hint: place code in OpeningFcn to populate axes2

% --- Executes on mouse press over axes background.
function axes2_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to axes2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

a = get(hObject, 'Currentpoint');

handles.clicks = handles.clicks + 1;
guidata(hObject, handles);

color_idx =  mod(handles.clicks,4);

for i = 1:size(handles.foci_list, 1)
    d_1(i) = ((a(1, 1)-handles.foci_list(i, 1))^2 + ((a(1, 2)-handles.foci_list(i,2))*10)^2)^0.5;
end
[R_1, j_1] = min(d_1);

if color_idx == 1
    plot( handles.foci_list(j_1, 1), handles.foci_list(j_1, 2), 'p', 'LineWidth',2 , 'MarkerEdgeColor','r','MarkerFaceColor','none','MarkerSize',15);
    handles.initiation_time = handles.foci_list(j_1, 1);
    handles.initiation_pos = handles.foci_list(j_1, 2);
    handles.initiation_mass = handles.length_list(2, find(handles.length_list(1,:) == handles.initiation_time)) / (2^(handles.n_oc-1)); %note that number of origins at initiation is fixed;
    
elseif color_idx == 2
    plot( handles.foci_list(j_1, 1), handles.foci_list(j_1, 2), 'o', 'LineWidth',2 , 'MarkerEdgeColor','b','MarkerFaceColor','none','MarkerSize',10);
    handles.termination_time = handles.foci_list(j_1, 1);
    handles.termination_pos = handles.foci_list(j_1, 2);
    handles.termination_mass = handles.length_list(2, find(handles.length_list(1,:) == handles.termination_time));
    
    plot( [handles.initiation_time handles.termination_time], [handles.initiation_pos handles.termination_pos], '-', 'LineWidth',1 , 'MarkerEdgeColor','b','MarkerFaceColor','none','MarkerSize',10, 'Color', [0.75 0.75 0.75]);
    
elseif color_idx == 3
    for i = 1:size(handles.birth_list, 1)
        d_3(i) = abs(a(1, 1)-handles.birth_list(i, 1));
    end
    [R_3, j_3] = min(d_3);
    plot( handles.birth_list(j_3, 1), handles.initiation_pos, 's', 'LineWidth',2 , 'MarkerEdgeColor','r','MarkerFaceColor','r','MarkerSize',10);
    handles.birth_time_m = handles.birth_list(j_3, 1);
    plot( [handles.birth_time_m handles.initiation_time], [handles.initiation_pos handles.initiation_pos], '-', 'LineWidth',1 , 'MarkerEdgeColor','b','MarkerFaceColor','none','MarkerSize',10, 'Color', [0.75 0.75 0.75]);

elseif color_idx == 0
    for i = 1:size(handles.division_list, 1)
        d_4(i) = abs(a(1, 1)-handles.division_list(i, 1));
    end
    [R_4, j_4] = min(d_4);
    plot( handles.division_list(j_4, 1), handles.termination_pos, 's', 'LineWidth',2 , 'MarkerEdgeColor','b','MarkerFaceColor','b','MarkerSize',10);
    handles.division_time = handles.division_list(j_4, 1);
    plot( [handles.termination_time handles.division_time], [handles.termination_pos handles.termination_pos], '-', 'LineWidth',1 , 'MarkerEdgeColor','b','MarkerFaceColor','none','MarkerSize',10, 'Color', [0.75 0.75 0.75]);
    
    cell_idx = find(handles.division_list == handles.division_time);
    
    cell_name_tmp = handles.cell_names{cell_idx,1};

    handles.cell_list.(cell_name_tmp).birth_time_m = handles.birth_time_m;
    
    handles.cell_list.(cell_name_tmp).initiation_time = handles.initiation_time;
    handles.cell_list.(cell_name_tmp).termination_time = handles.termination_time;
    
    handles.cell_list.(cell_name_tmp).initiation_mass = handles.initiation_mass;
    handles.cell_list.(cell_name_tmp).termination_mass = handles.termination_mass;
    
    guidata(hObject, handles);
end
                                          
guidata(hObject, handles);


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
cell_name_tmp = handles.cell_names;
cell_list = handles.cell_list;

save(handles.save_name,'cell_list');
saveas(handles.axes2,handles.save_name_png,'png');

guidata(hObject, handles);


% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles.channle_idx = handles.channle_idx-1;

cla;
[handles.length_list, handles.foci_list, handles.birth_list, handles.division_list, handles.cell_list, handles.cell_names, handles.save_name, handles.save_name_png, handles.display_name] ...
    = plot_channel(handles.dir_name, handles.cell_data, handles.px_to_mu, handles.IW_thr, handles.fnames_sort, handles.channels, handles.channle_idx, handles.xlim_max, handles.ylim_max, handles.time_int);
set(handles.display, 'String' , handles.display_name );

handles.clicks = 0;
handles.initiation_time = [];
handles.termination_time = [];
handles.division_time = [];

handles.birth_time_m = [];
handles.initiation_mass = [];
handles.termination_mass = [];

guidata(hObject, handles);


% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles.channle_idx = handles.channle_idx+1;

cla;
[handles.length_list, handles.foci_list, handles.birth_list, handles.division_list, handles.cell_list, handles.cell_names, handles.save_name, handles.save_name_png, handles.display_name] ...
    = plot_channel(handles.dir_name, handles.cell_data, handles.px_to_mu, handles.IW_thr, handles.fnames_sort, handles.channels, handles.channle_idx, handles.xlim_max, handles.ylim_max, handles.time_int);
set(handles.display, 'String' , handles.display_name );

handles.clicks = 0;
handles.initiation_time = [];
handles.termination_time = [];
handles.division_time = [];

handles.birth_time_m = [];
handles.initiation_mass = [];
handles.termination_mass = [];

guidata(hObject, handles);


% --- Executes during object deletion, before destroying properties.
function axes2_DeleteFcn(hObject, eventdata, handles)
% hObject    handle to axes2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
