clear
clc

files = dir('D:\Hao\SR_data\HCP_data\HCP_paired\T1w\*MPR1.mat');
len = length(files);

%%
dim = 3;
crop_size = 0;
crop_slice = 1;                     % must be an even number or 1
counter_training=0;
counter_validation=0;
counter_testing=0;
gap=18;
time_rot = 2;
time_stay = 5;
in_plane_rot = 5;
through_plane_rot = 5;
etl = 80;


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
    load(strcat('D:\Hao\SR_data\HCP_data\HCP_paired\T1w\',files(num).name));
    strcat('file:',num2str(num),'/',num2str(len))
    
    if mask_evaluation(num)==1
        counter_testing=counter_testing+1
        if evaluation_selected_mask(counter_testing)==1
            counter_1=counter_1+1;
            if mod(counter_1,3)==2
                [HRGT, LR]=MA_generate_3d_HCP(rot90(IMG1),crop_size,crop_slice,gap,time_rot,time_stay,in_plane_rot,through_plane_rot,etl); 
                filename2=strcat('D:\Hao\SR_data\HCP_data\MAR_',num2str(dim),'d_ax_',num2str(in_plane_rot),'-',num2str(through_plane_rot),'_',num2str(gap),'TR_',num2str(crop_size),'x',num2str(crop_slice),'\evaluation\data',num2str(counter_testing),'.mat');
                save(filename2, 'HRGT', 'LR', '-v7.3');
            end
        end
    elseif mask_validation(num)==1
        counter_validation=counter_validation+1
        if validation_selected_mask(counter_validation)==1
            counter_2=counter_2+1;
            if mod(counter_2,3)==2
                [HRGT, LR]=MA_generate_3d_HCP(rot90(IMG1),crop_size,crop_slice,gap,time_rot,time_stay,in_plane_rot,through_plane_rot,etl); 
                filename2=strcat('D:\Hao\SR_data\HCP_data\MAR_',num2str(dim),'d_ax_',num2str(in_plane_rot),'-',num2str(through_plane_rot),'_',num2str(gap),'TR_',num2str(crop_size),'x',num2str(crop_slice),'\validation\data',num2str(counter_validation),'.mat');
                save(filename2, 'HRGT', 'LR', '-v7.3');
            end
        end
    else
        counter_training=counter_training+1
        if training_selected_mask(counter_training)==1
            counter_3=counter_3+1;
            if mod(counter_3,3)==2
                [HRGT, LR]=MA_generate_3d_HCP(rot90(IMG1),crop_size,crop_slice,gap,time_rot,time_stay,in_plane_rot,through_plane_rot,etl); 
                filename2=strcat('D:\Hao\SR_data\HCP_data\MAR_',num2str(dim),'d_ax_',num2str(in_plane_rot),'-',num2str(through_plane_rot),'_',num2str(gap),'TR_',num2str(crop_size),'x',num2str(crop_slice),'\training\data',num2str(counter_training),'.mat');
                save(filename2, 'HRGT', 'LR', '-v7.3');
            end
        end
    end
    
end


%%
% clear
% clc
% files1 = dir('data*.mat');
% files2 = dir('.\ref\ref*.mat');
% len = length(files1);
% for num=1:len
%     files1(num).name
%     files2(num).name
%     load(files1(num).name);
%     load(strcat('.\ref\',files2(num).name),'REF');
%     filename=strcat('D:\Hao\SR_data\HCP_data\2x1_folds_3d_downsize_sag_128x1_ref\training\',files1(num).name);
%     save(filename,'HRGT','LR','REF','-v7.3');
% end
%             
% %%
% sl=700;
% figure;
% subplot(1,3,1);imagesc(HRGT(:,:,:,sl),[0 0.3]);colormap(gray);axis image;title('HRGT');
% subplot(1,3,2);imagesc(REF(:,:,:,sl),[0 0.3]);colormap(gray);axis image;title('REF');
% subplot(1,3,3);imagesc(LR(:,:,:,sl),[0 0.3]);colormap(gray);axis image;title('LR');