clc; clear; close all;

raw_data_root_dir = "D:\dataset\origin";
output_root_dir = "D:\dataset\processed_windowed_w512_s256";
window_size = 512;
window_slide = 256;

for subject = 1 : 20
    disp("Processing subject " + subject + "...");
    raw_data_file_path = raw_data_root_dir + "\s" + num2str(subject) + ".mat";
    
  	disp("Loading 'adjusted_class'...");
    load(raw_data_file_path, "adjusted_class");
    disp("Loading 'adjusted_repetition'...");
    load(raw_data_file_path, "adjusted_repetition");
    disp("Loading 'emg_extensors'...")
    load(raw_data_file_path, "emg_extensors");  % Shape [length, along_forearm = 8, across_forearm = 8]
    disp("Loading 'emg_flexors'...")
    load(raw_data_file_path, "emg_flexors");  % Shape [length, along_forearm = 8, across_forearm = 8]
    
    disp("Removing the signal segments of gesture 0 (rest) from 'emg_extensors'...");
    emg_extensors(adjusted_class == 0, :, :) = [];
    disp("Removing the signal segments of gesture 0 (rest) from 'emg_flexors'...");
    emg_flexors(adjusted_class == 0, :, :) = [];
    disp("Removing the signal segments of gesture 0 (rest) from 'adjusted_repetition'...");
    adjusted_repetition(adjusted_class == 0, :) = [];
    disp("Removing the signal segments of gesture 0 (rest) from 'adjusted_class'...");
    adjusted_class(adjusted_class == 0, :) = [];
    
    for fold = 1 : 5
        disp("Processing fold: " + num2str(fold));
        
        disp("Extracting class labels and EMG signals for the current fold (concatenating extensor and flexor channels)...");
        fold_adjusted_class = adjusted_class(adjusted_repetition == fold, :);
        fold_emg_data = cat(4, ...
            emg_extensors(adjusted_repetition == fold, :, :), ...
            emg_flexors(adjusted_repetition == fold, :, :));
        
        for class = 1 : 65
            disp("Processing class: " + num2str(class));
            
            disp("Extracting samples and corresponding EMG signals for the current gesture class...");
            current_class_labels = fold_adjusted_class(fold_adjusted_class == class, :);
            current_class_emg = fold_emg_data(fold_adjusted_class == class, :, :, :);

            idx = 0;
            for st = 1 : window_slide : size(current_class_labels, 1)
                idx = idx + 1;
                if st + window_size - 1 > size(current_class_labels, 1)
                    break;
                end
                
                disp("Extracting a single sliding window sample and rearranging dimensions to [2, time, height, width]...");
                window_emg_sample = current_class_emg(st : st + window_size - 1, :, :, :);  % [window_size, height, width, 2]
                window_emg_sample = permute(window_emg_sample, [4, 1, 2, 3]);  % [2, window_size, height, width]
                
                disp("Constructing output path for the current EMG sample (processed, windowed, per subject/fold/class)...");
                output_dir = output_root_dir + "\s" + num2str(subject) + "\fold" + num2str(fold) + "\class_" + num2str(class);
                if ~exist(output_dir, 'dir')
                    mkdir(output_dir);
                end
                
                disp("Opening file for writing binary float32 data...");
                output_file_path = output_dir + "\sample" + sprintf("%04d", idx) + ".bin";
                fp = fopen(output_file_path, "w");
                if fp == -1
                    error("Failed to open output file: " + output_file_path);
                end
                
                disp("Reordering dimensions to [width, height, time, 2] for row-major (C-style) binary storage...");
                emg_for_storage = permute(window_emg_sample, [4, 3, 2, 1]);
                
                disp("Writing flattened tensor as float32...");
                fwrite(fp, emg_for_storage(:), "float32");
                fclose(fp);
            end
        end
    end
end