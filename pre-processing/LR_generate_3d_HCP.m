function [HRGT,LR] = LR_generate_3d_HCP(IMG,in_plane_scale_factor,through_plane_scale_factor,zerofilling_cut_switch,crop_size,crop_slice,test_data,switch_2d_3d,eliptical)
%LR_GENERATE 此处显示有关此函数的摘要
%   此处显示详细说明
    
%     for i=1:size(IMG,3)
%         a=IMG(:,:,i);
%         standard_dev(i)=std(a(:));
%         average(i)=mean(a(:));
%     end
%     mean_std=mean(standard_dev);
%     mean_ave=mean(average);
%     for i=1:size(IMG,3)-2
%         if mean(standard_dev(i:i+2))<mean_std/10 && mean(average(i:i+2))<mean_ave/10
%             if mod(i+2,through_plane_scale_factor)==0
%                 IMG1=rot90(rot90(permute(IMG(:,:,1:i+2),[2,1,3])));
%             elseif mod(i+1,through_plane_scale_factor)==0
%                 IMG1=rot90(rot90(permute(IMG(:,:,1:i+1),[2,1,3])));
%             else
%                 IMG1=rot90(rot90(permute(IMG(:,:,1:i),[2,1,3])));
%             end
%             break;
%         end
%     end
    

%     IMG1 = IMG(:,:,floor(size(IMG,3)/(2*through_plane_scale_factor))*2*through_plane_scale_factor);
    IMG1=IMG(:,:,1:floor(size(IMG,3)/through_plane_scale_factor)*through_plane_scale_factor);
    [dim1,dim2,dim3]=size(IMG1);
%     kspace = zeros(dim1,dim2,dim3);
    
%    IMG4=IMG2(to(num):bo(num),le(num):ri(num),:);
%     [a,b]=ssim(IMG4(:,:,slc),IMG3(:,:,slc));
%     p = psnr(IMG4(:,:,slc),IMG3(:,:,slc));
%     figure;set(gcf,'Position',get(0,'ScreenSiz'));% title({strcat(num2str(scale_factor),32,'folds,',32,'slice',32,num2str(slc))})
%     subplot(1,2,1); imagesc(IMG3(:,:,slc),[0 WW]);colormap(gray);axis image;title('HRGT');
%     subplot(1,2,2); imagesc(IMG4(:,:,slc),[0 WW]);colormap(gray);axis image;title('LR');xlabel({strcat('SSIM:',32,num2str(a)),strcat('PSNR:',32,num2str(p))});
      
%%    
    IMG1=IMG1/max(IMG1(:));
%     kspace=fftn(IMG1);
    kspace=fftshift(fft(ifftshift(IMG1,1),[],1),1);
    kspace=fftshift(fft(ifftshift(kspace,2),[],2),2);
    kspace=fftshift(fft(ifftshift(kspace,3),[],3),3);
    if zerofilling_cut_switch==1
%         kspace(ceil(dim1*0.5/in_plane_scale_factor)+1:floor(dim1*(in_plane_scale_factor-0.5)/in_plane_scale_factor),:,:)=0;
%         kspace(:,ceil(dim2*0.5/in_plane_scale_factor)+1:ceil(dim2*(in_plane_scale_factor-0.5)/in_plane_scale_factor),:)=0;
%         kspace(:,:,ceil(dim3*0.5/through_plane_scale_factor)+1:ceil(dim3*(in_plane_scale_factor-0.5)/through_plane_scale_factor))=0;
%         IMG2=ifftn(kspace);
        kspace_cut = zeros(size(kspace));
        kspace_cut(dim1*0.5*(in_plane_scale_factor-1)/in_plane_scale_factor+1:dim1*0.5*(in_plane_scale_factor+1)/in_plane_scale_factor,dim2*0.5*(in_plane_scale_factor-1)/in_plane_scale_factor+1:dim2*0.5*(in_plane_scale_factor+1)/in_plane_scale_factor,dim3*0.5*(through_plane_scale_factor-1)/through_plane_scale_factor+1:dim3*0.5*(through_plane_scale_factor+1)/through_plane_scale_factor)=kspace(dim1*0.5*(in_plane_scale_factor-1)/in_plane_scale_factor+1:dim1*0.5*(in_plane_scale_factor+1)/in_plane_scale_factor,dim2*0.5*(in_plane_scale_factor-1)/in_plane_scale_factor+1:dim2*0.5*(in_plane_scale_factor+1)/in_plane_scale_factor,dim3*0.5*(through_plane_scale_factor-1)/through_plane_scale_factor+1:dim3*0.5*(through_plane_scale_factor+1)/through_plane_scale_factor);
        IMG2 = fftshift(ifft(ifftshift(kspace_cut,1),[],1),1);
        IMG2 = fftshift(ifft(ifftshift(IMG2,2),[],2),2);
        IMG2 = fftshift(ifft(ifftshift(IMG2,3),[],3),3);
        IMG2(:,:,:)=sqrt(real(IMG2(:,:,:)).^2+imag(IMG2(:,:,:)).^2);
        IMG2=IMG2/max(IMG2(:));    
    else
%         kspace_0=fftshift(kspace);
        kspace_cut=kspace(dim1*0.5*(in_plane_scale_factor-1)/in_plane_scale_factor+1:dim1*0.5*(in_plane_scale_factor+1)/in_plane_scale_factor,dim2*0.5*(in_plane_scale_factor-1)/in_plane_scale_factor+1:dim2*0.5*(in_plane_scale_factor+1)/in_plane_scale_factor,dim3*0.5*(through_plane_scale_factor-1)/through_plane_scale_factor+1:dim3*0.5*(through_plane_scale_factor+1)/through_plane_scale_factor);
%         IMG2=ifftn(fftshift(kspace_cut));
        mask=ones(size(kspace_cut));
        if eliptical==1
            for i=1:size(kspace_cut,2)
                for j=1:size(kspace_cut,3)
                    if (i-size(kspace_cut,2)/2+0.5)^2/(size(kspace_cut,2)/2)^2+(j-size(kspace_cut,3)/2+0.5)^2/(size(kspace_cut,3)/2)^2>1
                        mask(:,i,j)=0;
                    end
                end
            end
        end
        kspace_cut=kspace_cut.*mask;
        IMG2 = fftshift(ifft(ifftshift(kspace_cut,1),[],1),1);
        IMG2 = fftshift(ifft(ifftshift(IMG2,2),[],2),2);
        IMG2 = fftshift(ifft(ifftshift(IMG2,3),[],3),3);
        IMG2(:,:,:)=sqrt(real(IMG2(:,:,:)).^2+imag(IMG2(:,:,:)).^2);
        IMG2=IMG2/max(IMG2(:));
%         figure;set(gcf,'Position',get(0,'ScreenSize'));
%         subplot(1,3,1);imagesc(abs(permute(kspace(ceil(200/in_plane_scale_factor),:,:),[2,3,1])),[0 100]),colormap(gray);axis image;
%         subplot(1,3,2);imagesc(abs(permute(kspace_0(ceil(200/in_plane_scale_factor),:,:),[2,3,1])),[0 100]),colormap(gray);axis image;
%         subplot(1,3,3);imagesc(abs(permute(kspace_cut(ceil(200/in_plane_scale_factor),:,:),[2,3,1])),[0 100]),colormap(gray);axis image;
        if zerofilling_cut_switch==2
            IMG2=imresize3(IMG2,[dim1,dim2,dim3],'cubic');
        end
    end

    
%%
% 2D generator
if switch_2d_3d == 0
    if test_data
        counter=1;
%     noise_threshold = 0.03;
%     noise_ratio = 0.9;
%     ave_signal = 0.1;
%     total_threshold = floor(64 * 64 * (noise_threshold * noise_ratio + ave_signal * (1 - noise_ratio)));

    

        stride = crop_size*3/4;
    
        for k=1:size(IMG2,3)-(crop_slice/through_plane_scale_factor-1)
            for i=0:(size(IMG1,1)-crop_size)/stride
                for j=0:(size(IMG1,2)-crop_size)/stride
                    counter;
                    HRGT(:,:,:,counter)=IMG1(i*stride+1:i*stride+crop_size,j*stride+1:j*stride+crop_size,(k-1)*through_plane_scale_factor+1:(k-1)*through_plane_scale_factor+crop_slice);
                    if zerofilling_cut_switch>0
                        LR(:,:,:,counter)=IMG2(i*stride+1:i*stride+crop_size,j*stride+1:j*stride+crop_size,(k-1)*through_plane_scale_factor+1:(k-1)*through_plane_scale_factor+crop_slice);
                    else
                        LR(:,:,:,counter)=IMG2(i*stride/in_plane_scale_factor+1:(i*stride+crop_size)/in_plane_scale_factor,j*stride/in_plane_scale_factor+1:(j*stride+crop_size)/in_plane_scale_factor,k:k+crop_slice/through_plane_scale_factor-1);
                    end
                    counter = counter + 1;
                end
            end
        end
    else
        HRGT=IMG1;
        LR=IMG2;
    end
else
    if test_data
        counter=1;
%     noise_threshold = 0.03;
%     noise_ratio = 0.9;
%     ave_signal = 0.1;
%     total_threshold = floor(64 * 64 * (noise_threshold * noise_ratio + ave_signal * (1 - noise_ratio)));

    

        stride = crop_size*1/2;
    else
        stride = crop_size;
    end  
    
        for k=0:(size(IMG1,3)-crop_size)/stride
            for i=0:(size(IMG1,1)-crop_size)/stride
                for j=0:(size(IMG1,2)-crop_size)/stride
                    counter;
                    HRGT(:,:,:,counter)=IMG1(i*stride+1:i*stride+crop_size,j*stride+1:j*stride+crop_size,k*stride+1:k*stride+crop_size);
                    if zerofilling_cut_switch
                        LR(:,:,:,counter)=IMG2(i*stride+1:i*stride+crop_size,j*stride+1:j*stride+crop_size,k*stride+1:k*stride+crop_size);
                    else
                        LR(:,:,:,counter)=IMG2(i*stride/in_plane_scale_factor+1:(i*stride+crop_size)/in_plane_scale_factor,j*stride/in_plane_scale_factor+1:(j*stride+crop_size)/in_plane_scale_factor,k*stride/in_plane_scale_factor+1:(k*stride+crop_size)/in_plane_scale_factor);
                    end
                    counter = counter + 1;
                end
            end
        end
  
end



end

