function [HRGT,LR] = MA_generate_3d_HCP(IMG1,crop_size,crop_slice,gap,time_rot,time_stay,in_plane_rot,through_plane_rot,etl)

    [dim1,dim2,dim3]=size(IMG1);
if in_plane_rot~=0 && through_plane_rot~=0
    IMG1=IMG1/max(IMG1(:));
    IMG1_R1 = imrotate3(IMG1,in_plane_rot/(time_rot+1),[0,0,1],'cubic','crop');
    IMG1_R2 = imrotate3(IMG1,2*in_plane_rot/(time_rot+1),[0,0,1],'cubic','crop');
    IMG1_R = imrotate3(IMG1,in_plane_rot,[0,0,1],'cubic','crop');
    IMG1_L1 = imrotate3(IMG1,-in_plane_rot/(time_rot+1),[0,0,1],'cubic','crop');
    IMG1_L2 = imrotate3(IMG1,-2*in_plane_rot/(time_rot+1),[0,0,1],'cubic','crop');
    IMG1_L = imrotate3(IMG1,-in_plane_rot,[0,0,1],'cubic','crop');

    if through_plane_rot>0
        IMG1_R1 = imrotate3(IMG1_R1,-through_plane_rot/(time_rot+1),[0,1,0],'cubic','crop');
        IMG1_R2 = imrotate3(IMG1_R2,-2*through_plane_rot/(time_rot+1),[0,1,0],'cubic','crop');
        IMG1_R = imrotate3(IMG1_R,-through_plane_rot,[0,1,0],'cubic','crop');
        IMG1_L1 = imrotate3(IMG1_L1,-through_plane_rot/(time_rot+1),[0,1,0],'cubic','crop');
        IMG1_L2 = imrotate3(IMG1_L2,-2*through_plane_rot/(time_rot+1),[0,1,0],'cubic','crop');
        IMG1_L = imrotate3(IMG1_L,-through_plane_rot,[0,1,0],'cubic','crop');
    end
    
%     figure;
%     subplot(3,3,2);imagesc(IMG1(:,:,100),[0 0.5]);colormap(gray);axis image;
%     subplot(3,3,4);imagesc(IMG1_R1(:,:,100),[0 0.5]);colormap(gray);axis image;
%     subplot(3,3,5);imagesc(IMG1_R2(:,:,100),[0 0.5]);colormap(gray);axis image;
%     subplot(3,3,6);imagesc(IMG1_R(:,:,100),[0 0.5]);colormap(gray);axis image;
%     subplot(3,3,7);imagesc(IMG1_L1(:,:,100),[0 0.5]);colormap(gray);axis image;
%     subplot(3,3,8);imagesc(IMG1_L2(:,:,100),[0 0.5]);colormap(gray);axis image;
%     subplot(3,3,9);imagesc(IMG1_L(:,:,100),[0 0.5]);colormap(gray);axis image;
    
    kspace1=fftshift(fft(ifftshift(IMG1,1),[],1),1);
    kspace1=fftshift(fft(ifftshift(kspace1,2),[],2),2);
    kspace1=reshape(fftshift(fft(ifftshift(kspace1,3),[],3),3),[dim1,dim2*dim3]);
    
    kspace_R1=fftshift(fft(ifftshift(IMG1_R1,1),[],1),1);
    kspace_R1=fftshift(fft(ifftshift(kspace_R1,2),[],2),2);
    kspace_R1=reshape(fftshift(fft(ifftshift(kspace_R1,3),[],3),3),[dim1,dim2*dim3]);
    
    kspace_R2=fftshift(fft(ifftshift(IMG1_R2,1),[],1),1);
    kspace_R2=fftshift(fft(ifftshift(kspace_R2,2),[],2),2);
    kspace_R2=reshape(fftshift(fft(ifftshift(kspace_R2,3),[],3),3),[dim1,dim2*dim3]);
    
    kspace_R=fftshift(fft(ifftshift(IMG1_R,1),[],1),1);
    kspace_R=fftshift(fft(ifftshift(kspace_R,2),[],2),2);
    kspace_R=reshape(fftshift(fft(ifftshift(kspace_R,3),[],3),3),[dim1,dim2*dim3]);
    
    kspace_L1=fftshift(fft(ifftshift(IMG1_L1,1),[],1),1);
    kspace_L1=fftshift(fft(ifftshift(kspace_L1,2),[],2),2);
    kspace_L1=reshape(fftshift(fft(ifftshift(kspace_L1,3),[],3),3),[dim1,dim2*dim3]);
    
    kspace_L2=fftshift(fft(ifftshift(IMG1_L2,1),[],1),1);
    kspace_L2=fftshift(fft(ifftshift(kspace_L2,2),[],2),2);
    kspace_L2=reshape(fftshift(fft(ifftshift(kspace_L2,3),[],3),3),[dim1,dim2*dim3]);
    
    kspace_L=fftshift(fft(ifftshift(IMG1_L,1),[],1),1);
    kspace_L=fftshift(fft(ifftshift(kspace_L,2),[],2),2);
    kspace_L=reshape(fftshift(fft(ifftshift(kspace_L,3),[],3),3),[dim1,dim2*dim3]);
    
    
    turn_left = 1;
    stay_left = 0;
    return_left = 0;
    stay_org = 0;
    turn_right = 0;
    stay_right = 0;
    return_right = 0;
    left=1;
    right=0;
    kspace2 = zeros(size(kspace1));
    center=dim2*dim3/2;
    index=gap/2;
    while center-index*etl>0
        if index==gap/2
            kspace2(:,center-index*etl+1:center+index*etl)=kspace1(:,center-index*etl+1:center+index*etl);
        end
        counter=index;
        if center-(index+0.5)*etl<0
            break
        end
        if turn_left==1
            kspace2(:,center-(index+0.5)*etl+1:center-index*etl)=kspace_L1(:,center-(index+0.5)*etl+1:center-index*etl);
            kspace2(:,center+index*etl+1:center+(index+0.5)*etl)=kspace_L1(:,center+index*etl+1:center+(index+0.5)*etl);
            turn_left=2;
            index=index+0.5;
        end
        counter=index;
        if center-(index+0.5)*etl<0
            break
        end
        if turn_left==2
            kspace2(:,center-(index+0.5)*etl+1:center-index*etl)=kspace_L2(:,center-(index+0.5)*etl+1:center-index*etl);
            kspace2(:,center+index*etl+1:center+(index+0.5)*etl)=kspace_L2(:,center+index*etl+1:center+(index+0.5)*etl);
            turn_left=0;
            stay_left=1;
            index=index+0.5;
        end
        counter=index;
        if center-(index+time_stay/2)*etl<0
            break
        end
        if stay_left>0
            kspace2(:,center-(index+time_stay/2)*etl+1:center-index*etl)=kspace_L(:,center-(index+time_stay/2)*etl+1:center-index*etl);
            kspace2(:,center+index*etl+1:center+(index+time_stay/2)*etl)=kspace_L(:,center+index*etl+1:center+(index+time_stay/2)*etl);
            index=index+time_stay/2;
            stay_left=0;
            return_left=1;
        end
        counter=index;
        if center-(index+0.5)*etl<0
            break
        end
        if return_left==1
            kspace2(:,center-(index+0.5)*etl+1:center-index*etl)=kspace_L2(:,center-(index+0.5)*etl+1:center-index*etl);
            kspace2(:,center+index*etl+1:center+(index+0.5)*etl)=kspace_L2(:,center+index*etl+1:center+(index+0.5)*etl);
            return_left=2;
            index=index+0.5;
        end
        counter=index;
        if center-(index+0.5)*etl<0
            break
        end
        if return_left==2
            kspace2(:,center-(index+0.5)*etl+1:center-index*etl)=kspace_L1(:,center-(index+0.5)*etl+1:center-index*etl);
            kspace2(:,center+index*etl+1:center+(index+0.5)*etl)=kspace_L1(:,center+index*etl+1:center+(index+0.5)*etl);
            return_left=0;
            stay_org=1;
            index=index+0.5;
        end
        counter=index;
        if center-(index+gap/2)*etl<0
            break
        end
        if stay_org>0
            kspace2(:,center-(index+gap/2)*etl+1:center-index*etl)=kspace1(:,center-(index+gap/2)*etl+1:center-index*etl);
            kspace2(:,center+index*etl+1:center+(index+gap/2)*etl)=kspace1(:,center+index*etl+1:center+(index+gap/2)*etl);
            index=index+gap/2;
            stay_org=0;
            if left==1
                left=0;
                right=1;
                turn_right=1;
            elseif right==1
                left=1;
                right=0;
                turn_left=1;
            end
        end
        counter=index;
        if center-(index+0.5)*etl<0
            break
        end
        if turn_right==1
            kspace2(:,center-(index+0.5)*etl+1:center-index*etl)=kspace_R1(:,center-(index+0.5)*etl+1:center-index*etl);
            kspace2(:,center+index*etl+1:center+(index+0.5)*etl)=kspace_R1(:,center+index*etl+1:center+(index+0.5)*etl);
            turn_right=2;
            index=index+0.5;
        end
        counter=index;
        if center-(index+0.5)*etl<0
            break
        end
        if turn_right==2
            kspace2(:,center-(index+0.5)*etl+1:center-index*etl)=kspace_R2(:,center-(index+0.5)*etl+1:center-index*etl);
            kspace2(:,center+index*etl+1:center+(index+0.5)*etl)=kspace_R2(:,center+index*etl+1:center+(index+0.5)*etl);
            turn_right=0;
            stay_right=1;
            index=index+0.5;
        end
        counter=index;
        if center-(index+time_stay/2)*etl<0
            break
        end
        if stay_right>0
            kspace2(:,center-(index+time_stay/2)*etl+1:center-index*etl)=kspace_R(:,center-(index+time_stay/2)*etl+1:center-index*etl);
            kspace2(:,center+index*etl+1:center+(index+time_stay/2)*etl)=kspace_R(:,center+index*etl+1:center+(index+time_stay/2)*etl);
            index=index+time_stay/2;
            stay_right=0;
            return_right=1;
        end
        counter=index;
        if center-(index+0.5)*etl<0
            break
        end
        if return_right==1
            kspace2(:,center-(index+0.5)*etl+1:center-index*etl)=kspace_R2(:,center-(index+0.5)*etl+1:center-index*etl);
            kspace2(:,center+index*etl+1:center+(index+0.5)*etl)=kspace_R2(:,center+index*etl+1:center+(index+0.5)*etl);
            return_right=2;
            index=index+0.5;
        end
        counter=index;
        if center-(index+0.5)*etl<0
            break
        end
        if return_right==2
            kspace2(:,center-(index+0.5)*etl+1:center-index*etl)=kspace_R1(:,center-(index+0.5)*etl+1:center-index*etl);
            kspace2(:,center+index*etl+1:center+(index+0.5)*etl)=kspace_R1(:,center+index*etl+1:center+(index+0.5)*etl);
            return_right=0;
            stay_org=1;
            index=index+0.5;
        end
        counter=index;
        if center-(index+gap/2)*etl<0
            break
        end
        if stay_org>0
            kspace2(:,center-(index+gap/2)*etl+1:center-index*etl)=kspace1(:,center-(index+gap/2)*etl+1:center-index*etl);
            kspace2(:,center+index*etl+1:center+(index+gap/2)*etl)=kspace1(:,center+index*etl+1:center+(index+gap/2)*etl);
            index=index+gap/2;
            stay_org=0;
            if left==1
                left=0;
                right=1;
                turn_right=1;
            elseif right==1
                left=1;
                right=0;
                turn_left=1;
            end
        end
    end
    if center-counter*etl>0
        kspace2(:,1:center-counter*etl)=kspace1(:,1:center-counter*etl);
        kspace2(:,center+counter*etl+1:center*2)=kspace1(:,center+counter*etl+1:center*2);
    end
    kspace2 = reshape(kspace2,[dim1,dim2,dim3]);
    IMG2 = fftshift(ifft(ifftshift(kspace2,1),[],1),1);
    IMG2 = fftshift(ifft(ifftshift(IMG2,2),[],2),2);
    IMG2 = fftshift(ifft(ifftshift(IMG2,3),[],3),3);
    IMG2(:,:,:)=sqrt(real(IMG2(:,:,:)).^2+imag(IMG2(:,:,:)).^2);
    IMG2=IMG2/max(IMG2(:));
else
    IMG2=IMG1;
end
    
%%
% 2D/3D generator
if crop_size>0
    counter=1;

    stride = crop_size*3/4;
    
    if crop_slice==1
        for k=1:size(IMG2,3)-(crop_slice-1)
            for index=0:(size(IMG1,1)-crop_size)/stride
                for j=0:(size(IMG1,2)-crop_size)/stride
                    counter;
                    HRGT(:,:,:,counter)=IMG1(index*stride+1:index*stride+crop_size,j*stride+1:j*stride+crop_size,k);
                    LR(:,:,:,counter)=IMG2(index*stride+1:index*stride+crop_size,j*stride+1:j*stride+crop_size,k);
                    counter = counter + 1;
                end
            end
        end
    else
        for k=1:size(IMG2,3)-(crop_slice-1)
            for index=0:(size(IMG1,1)-crop_size)/stride
                for j=0:(size(IMG1,2)-crop_size)/stride
                    counter;
                    HRGT(:,:,:,counter)=IMG1(index*stride+1:index*stride+crop_size,j*stride+1:j*stride+crop_size,k:k-1+crop_slice);
                    LR(:,:,:,counter)=IMG2(index*stride+1:index*stride+crop_size,j*stride+1:j*stride+crop_size,k:k-1+crop_slice);
                    counter = counter + 1;
                end
            end
        end
    end
else
    if crop_slice==1
        HRGT = IMG1;
        LR = IMG2;
    else
        counter=1;
        for k=1:(size(IMG1,3)-crop_slice+1)
            counter;
            HRGT(:,:,:,counter)=IMG1(:,:,k:k-1+crop_slice);
            LR(:,:,:,counter)=IMG2(:,:,k:k-1+crop_slice);
            counter = counter + 1;
        end
    end
end



end

