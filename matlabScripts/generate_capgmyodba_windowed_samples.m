clc; clear; close all;

raw_data_root_dir = "D:\dataset\7210397\origin";
output_root_dir = "D:\dataset\7210397\processed_windowed_w200_s100";
window_size = 200;
window_slide = 100;
num_subjects = 18;
num_gestures = 8;
num_trials = 10;
num_rows = 8;
num_cols = 16;
num_trails_per_fold = 2;

for subject = 1 : num_subjects
    for gesture = 1 : num_gestures
        for trial = 1 : num_trials
            raw_data_file_path = raw_data_root_dir + "\dba-s" + num2str(subject) + sprintf("\\%03d-%03d-%03d.mat", subject, gesture, trial);
            disp("Processing file: " + raw_data_file_path);
            load(raw_data_file_path, "data");
            
            fold = ceil(trial / num_trails_per_fold);
            class = gesture;
            output_dir = output_root_dir + "\s" + num2str(subject) + "\fold" + num2str(fold) + "\class_" + num2str(class);
            if ~exist(output_dir, 'dir')
                mkdir(output_dir);
            end

            data_length = size(data, 1);
            window_id = 0;
            for start_at = 1 : window_slide : data_length
                window_id = window_id + 1;
                end_at = start_at + window_size - 1;
                if end_at > data_length
                    break
                end
                
                window_emg_sample = data(start_at : end_at, :);  % [window_size, num_channels]
                window_emg_sample = reshape(window_emg_sample, [window_size, num_rows, num_cols]);
                
                output_file_path = output_dir + "\sample" + sprintf("%04d", window_id) + ".bin";
                fp = fopen(output_file_path, "w");
                if fp == -1
                    error("Failed to open output file: " + output_file_path);
                end
                
                emg_for_storage = permute(window_emg_sample, [3, 2, 1]);
                fwrite(fp, emg_for_storage(:), "float32");
                fclose(fp);
            end
        end
    end
end