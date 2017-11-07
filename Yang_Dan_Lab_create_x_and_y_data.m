clear all; close all; clc;

%% Instructions
% Download the crcns-pvc2 dataset from:
% https://crcns.org/data-sets/vc/pvc-2/about
% and place the content in the same folder as this script or in a subfolder

% Correct this path
data_path = 'crcns-pvc2/1D_white_noise/Spike_and_Log_Files/';
% This should resolve all other paths assuming that the data and all
% included matlab files are found in subdirectories
addpath(genpath(pwd))

%% Parameters
% Used for calculating and plotting the STA and two STC filters
win_size = 16;
low = 0.1;
high = 0.9;
cmap = [linspace(low, high, 50), linspace(high, 1, 50); 
        linspace(low, high, 50), linspace(high, low, 50); 
        linspace(1, high, 50), linspace(high, low, 50);]';

%% Initialization
% Create a data subdirectory in none exist
save_folder = 'data';
if exist(save_folder, 'dir') ~= 7
  mkdir(save_folder)
end
      
%% X data

stimuli = load('msq1D.mat', 'msq1D');
stimuli = stimuli.('msq1D');

% Create a separate stimulus matrix by sliding a window over the original
% one. This is used for calculating the STA and two STC filters for each
% neuron.
x_win = zeros(size(stimuli, 1)-win_size+1, size(stimuli, 2)*win_size);
for win_id = 1:size(x_win, 1)
  x_win(win_id, :) = reshape(stimuli(win_id:win_id+win_size-1, :), ...
                             1, win_size*size(stimuli, 2));
end

%% Y data

fig_h = figure(1);
pos = get(fig_h,'position');
set(fig_h,'position',[0 pos(2) pos(3)*2.5 pos(4)])

dir_list = dir(data_path);

for list_id = 3:length(dir_list)
  
  if dir_list(list_id).isdir
    
    x = [];
    y = [];
    ste = [];
    disp(dir_list(list_id).name)
    file_list = dir([data_path, dir_list(list_id).name]);
    
    for file_id = 3:length(file_list)
      
      if strcmp(file_list(file_id).name(end-8:end), 'msq1D.sa0') || ...
          strcmp(file_list(file_id).name(end-8:end), 'msq1d.sa0')
        
        % Read the frame rate from the log file
        fid = fopen([file_list(file_id).name(1:end-3), 'log']);
        text = textscan(fid, '%s');
        fclose(fid);
        frame_rate_line = ...
          textscan(text{1}{17}, '%s%f', 'whitespace', text{1}{17}(15));
        if not(strcmp(frame_rate_line{1}{1}, 'FrameRate'))
          frpintf('WARNING FRAME RATE NOT FOUND')
        end
        frame_rate = frame_rate_line{2};
          
        
        % Read spike data
        dt = 1e4/frame_rate; % in 0.1ms
        
        fprintf('Reading file: %s ...', file_list(file_id).name);

        spike_times = fget_spk(file_list(file_id).name);
        time_bin_edges = linspace(0, size(stimuli, 1)*dt, size(stimuli, 1)+1);
        y_tmp = histcounts(spike_times, time_bin_edges);

        x = [x; stimuli];
        y = [y; y_tmp'];

        for spike_count = 1:max(y_tmp)
          ste = [ste; x_win(y_tmp(win_size:end) == spike_count, :)];
        end

        fprintf(' DONE!\n')
        
      end
      
    end
    
    % Calculating the STA and the STC filters
    sta = mean(ste, 1);
    stc = cov(ste);
    avg_firing_rate = mean(y)*frame_rate;
    [vec, val] = eig(stc);
    
    % STA non-linearity
    z_sta = x_win*sta';
    z_sta_spike = ste*sta';
    [n_sta, edges_sta] = histcounts(z_sta, 11);
    [n_sta_spike, edges_sta] = histcounts(z_sta_spike, edges_sta);

    % STC non-linearities
    top2_vec = vec(:, end-1:end);    
    z_stc = x_win*top2_vec;
    z_stc_spike = ste*top2_vec;
    [n1, edges1] = histcounts(z_stc(:, 2), 11);
    [n1_spike, edges1] = histcounts(z_stc_spike(:, 2), edges1);
    [n2, edges2] = histcounts(z_stc(:, 1), 11);
    [n2_spike, edges2] = histcounts(z_stc_spike(:, 1), edges2);
    
    % Plotting the filters and their non-linearities
    clf()
    subplot(2, 4, 1)
    c_lim = max(abs([max(max(sta)), min(min(sta))]));
    imagesc(reshape(sta, win_size, size(stimuli, 2))', [-c_lim, c_lim])
    title(sprintf('STA, (mean rate: %2.2f Hz)', avg_firing_rate))
    set(gca,'Ydir','normal')
    ylabel('Spatial dim')
    xlabel('time')
    subplot(2, 4, 2)
    c_lim = max(abs([max(max(vec(:, end))), min(min(vec(:, end)))]));
    imagesc(reshape(top2_vec(:, end), win_size, size(stimuli, 2))', [-c_lim, c_lim])
    title('Largest eigen vector')
    set(gca,'Ydir','normal')
    subplot(2, 4, 3)
    c_lim = max(abs([max(max(vec(:, end-1))), min(min(vec(:, end-1)))]));
    imagesc(reshape(top2_vec(:, end-1), win_size, size(stimuli, 2))', [-c_lim, c_lim])
    title('2nd largest eigen vector')
    set(gca,'Ydir','normal')
    subplot(2, 4, 4)
    plot(diag(val), 'ko')
    title('STC eigen values')
    subplot(2, 4, 5)
    plot(edges_sta(1:end-1) + (edges_sta(2)-edges_sta(1))/2, avg_firing_rate*n_sta_spike./n_sta, 'ko')
    xlabel('Similarity score (z)')
    ylabel('Expected spike count')
    subplot(2, 4, 6)
    plot(edges1(1:end-1) + (edges1(2)-edges1(1))/2, avg_firing_rate*n1_spike./n1, 'ko')
    subplot(2, 4, 7)
    plot(edges2(1:end-1) + (edges2(2)-edges2(1))/2, avg_firing_rate*n2_spike./n2, 'ko')
    
    colormap(cmap)

    % Save a copy of the figure
    save_name = [save_folder, '/', dir_list(list_id).name, '.png'];
    saveas(gcf(), save_name)
    
    % Save the data
    x_labels = {'Time (ms)', 'Spatial dim'};
    x_ticks = {[], []};
    name = dir_list(list_id).name;
    origin = 'measurement';
    dt_ms = dt / 10;
    save_name = [save_folder, '/', dir_list(list_id).name, '.mat'];
    save(save_name, 'x', 'x_labels', 'x_ticks', ...
      'y', 'name', 'origin', 'dt_ms', '-v7')
    
    pause(0.001)
    
  end
  
end