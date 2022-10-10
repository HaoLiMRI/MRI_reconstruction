clear
clc

files = dir('D:\Hao\SR_data\HCP_data\HCP_paired\T2w\*SPC1.mat');
len = length(files);

%%
in_plane_scale_factor = 2;
through_plane_scale_factor = 2;
zerofilling_cut_switch = 0;         % Kspace zero-filling or cutting. Cutting decrease the size of LR image. 0: cutting, 1: zero-filling, 2: cubic interpolation.
switch_2d_3d = 0;                   % 0: 2d, 1: 3d
dim = 3;
crop_size = 128;
crop_slice = 6;                     % must be an even number or 1
eliptical = 0;                      % eliptical kspace. 0: off, 1: on.
counter_training=0;
counter_validation=0;
counter_testing=0;

counter_1=0;
counter_2=0;
counter_3=0;

load('D:\Hao\SR_data\mask_evaluation.mat');
load('D:\Hao\SR_data\mask_validation.mat');
load('D:\Hao\SR_data\training_selected_mask.mat');
load('D:\Hao\SR_data\validation_selected_mask.mat');
load('D:\Hao\SR_data\evaluation_selected_mask.mat');

%%
for num=1:len
    strcat('file:',num2str(num),'/',num2str(len))
    
    if mask_evaluation(num)==1
        test_data = 1;
        counter_testing=counter_testing+1
        if evaluation_selected_mask(counter_testing)==1
            counter_1=counter_1+1;
            %if mod(counter_1,3)==2
                load(strcat('D:\Hao\SR_data\HCP_data\HCP_paired\T2w\',files(num).name));
                [HRGT, LR]=LR_generate_3d_HCP(rot90(rot90(permute(IMG1,[3,2,1]))),in_plane_scale_factor,through_plane_scale_factor,zerofilling_cut_switch,crop_size,crop_slice,test_data,switch_2d_3d,eliptical); 
                if switch_2d_3d
                    filename2=strcat('D:\Hao\SR_data\HCP_data\',num2str(in_plane_scale_factor),'x',num2str(through_plane_scale_factor),'_folds_',num2str(dim),'d_downsize_sag_',num2str(crop_size),'x',num2str(crop_slice/through_plane_scale_factor),'\Evaluation\data',num2str(counter_testing),'.mat');
                else
                    filename2=strcat('D:\Hao\SR_data\HCP_data\',num2str(in_plane_scale_factor),'x',num2str(through_plane_scale_factor),'_folds_',num2str(dim),'d_downsize_sag_',num2str(crop_size),'x',num2str(crop_slice/through_plane_scale_factor),'\Evaluation\data',num2str(counter_testing),'.mat');
                end
                save(filename2, 'HRGT', 'LR', '-v7.3');
            %end
        end
    elseif mask_validation(num)==1
        test_data = 1;
        counter_validation=counter_validation+1
        if validation_selected_mask(counter_validation)==1
            counter_2=counter_2+1;
            %if mod(counter_2,3)==2
                load(strcat('D:\Hao\SR_data\HCP_data\HCP_paired\T2w\',files(num).name));
                [HRGT, LR]=LR_generate_3d_HCP(rot90(rot90(permute(IMG1,[3,2,1]))),in_plane_scale_factor,through_plane_scale_factor,zerofilling_cut_switch,crop_size,crop_slice,test_data,switch_2d_3d,eliptical); 
                if switch_2d_3d
                    filename2=strcat('D:\Hao\SR_data\HCP_data\',num2str(in_plane_scale_factor),'x',num2str(through_plane_scale_factor),'_folds_',num2str(dim),'d_downsize_sag_',num2str(crop_size),'x',num2str(crop_slice/through_plane_scale_factor),'\Validation\data',num2str(counter_validation),'.mat');
                else
                    filename2=strcat('D:\Hao\SR_data\HCP_data\',num2str(in_plane_scale_factor),'x',num2str(through_plane_scale_factor),'_folds_',num2str(dim),'d_downsize_sag_',num2str(crop_size),'x',num2str(crop_slice/through_plane_scale_factor),'\Validation\data',num2str(counter_validation),'.mat');
                end
                save(filename2, 'HRGT', 'LR', '-v7.3');
            %end
        end
    else
        test_data = 1;
        counter_training=counter_training+1
        if training_selected_mask(counter_training)==1
            counter_3=counter_3+1;
            %if mod(counter_3,3)==2
                load(strcat('D:\Hao\SR_data\HCP_data\HCP_paired\T2w\',files(num).name));
                [HRGT, LR]=LR_generate_3d_HCP(rot90(rot90(permute(IMG1,[3,2,1]))),in_plane_scale_factor,through_plane_scale_factor,zerofilling_cut_switch,crop_size,crop_slice,test_data,switch_2d_3d,eliptical); 
                if switch_2d_3d
                    filename2=strcat('D:\Hao\SR_data\HCP_data\',num2str(in_plane_scale_factor),'x',num2str(through_plane_scale_factor),'_folds_',num2str(dim),'d_downsize_sag_',num2str(crop_size),'x',num2str(crop_slice/through_plane_scale_factor),'\Training\data',num2str(counter_training),'.mat');
                else
                    filename2=strcat('D:\Hao\SR_data\HCP_data\',num2str(in_plane_scale_factor),'x',num2str(through_plane_scale_factor),'_folds_',num2str(dim),'d_downsize_sag_',num2str(crop_size),'x',num2str(crop_slice/through_plane_scale_factor),'\Training\data',num2str(counter_training),'.mat');
                end
                save(filename2, 'HRGT', 'LR', '-v7.3');
            %end
        end
    end
    
end