s = {};
s{1,1} = 'image_id';s{1,2} = 'ACC';s{1,3} ='DICE';s{1,4} = 'IOU';s{1,5}='REC';s{1,6} ='SEN';

for kk=1:num
    
data1=load(['data path',num2str(kk),'.mat']);
data2=im2uint16(data1.predict);
data3=double(data2(:,:,:,1)/65535);
data4 = [data3(:,:,1) data3(:,:,2) data3(:,:,3) data3(:,:,4) data3(:,:,5) data3(:,:,6) data3(:,:,7) data3(:,:,8)...
         data3(:,:,9) data3(:,:,10) data3(:,:,11) data3(:,:,12) data3(:,:,13) data3(:,:,14) data3(:,:,15) data3(:,:,16)...
         data3(:,:,17) data3(:,:,18) data3(:,:,19) data3(:,:,20) data3(:,:,21) data3(:,:,22) data3(:,:,23) data3(:,:,24)...
         data3(:,:,25) data3(:,:,26) data3(:,:,27) data3(:,:,28) data3(:,:,29) data3(:,:,30) data3(:,:,31) data3(:,:,32)];

label1=load(['label path','label',num2str(kk),'.mat']);
label2=label1.label0;
label3=double(label2);
label4 = [label3(:,:,1) label3(:,:,2) label3(:,:,3) label3(:,:,4) label3(:,:,5) label3(:,:,6) label3(:,:,7) label3(:,:,8)...
         label3(:,:,9) label3(:,:,10) label3(:,:,11) label3(:,:,12) label3(:,:,13) label3(:,:,14) label3(:,:,15) label3(:,:,16)...
         label3(:,:,17) label3(:,:,18) label3(:,:,19) label3(:,:,20) label3(:,:,21) label3(:,:,22) label3(:,:,23) label3(:,:,24)...
         label3(:,:,25) label3(:,:,26) label3(:,:,27) label3(:,:,28) label3(:,:,29) label3(:,:,30) label3(:,:,31) label3(:,:,32)];

% figure()
% subplot(211);imshow(data4)
% subplot(212);imshow(label4)
p = caculate(data4,label4);

s{kk + 1,1} = num2str(kk);
s{kk + 1,2} = p(1);
s{kk + 1,3} = p(2);
s{kk + 1,4} = p(3);
s{kk + 1,5} = p(4);
s{kk + 1,6} = p(5);
end

s{17,1} = 'ave';
s{17,2} = roundn(sum(cell2mat(s(2:16,2)))/15,-4);
s{17,3} = roundn(sum(cell2mat(s(2:16,3)))/15,-4);
s{17,4} = roundn(sum(cell2mat(s(2:16,4)))/15,-4);
s{17,5} = roundn(sum(cell2mat(s(2:16,5)))/15,-4);
s{17,6} = roundn(sum(cell2mat(s(2:16,6)))/15,-4);

s{18,1} = 'std';
s{18,2} = roundn(std(cell2mat(s(2:16,2))),-4);
s{18,3} = roundn(std(cell2mat(s(2:16,3))),-4);
s{18,4} = roundn(std(cell2mat(s(2:16,4))),-4);
s{18,4} = roundn(std(cell2mat(s(2:16,5))),-4);
s{18,4} = roundn(std(cell2mat(s(2:16,6))),-4);

local_dir = 'outdir path';
save_dir = strcat(local_dir , "value.xls");
xlswrite(save_dir, s);





function x = caculate(l_pic, gt_pic)
    TP = 0;
    TN = 0;
    FP = 0;
    FN = 0;
    
     
    f_l_pic = double(ones(size(l_pic))) - l_pic;
    f_gt_pic = double(ones(size(l_pic))) - gt_pic;  
        
    s_TP = l_pic .* gt_pic;       TP = sum(sum(s_TP));
    s_FN = f_l_pic .* gt_pic;     FN = sum(sum(s_FN));
    s_FP = l_pic .* f_gt_pic;     FP = sum(sum(s_FP));
    s_TN = f_l_pic .* f_gt_pic;   TN = sum(sum(s_TN));


    if TP == 0
        ACC = 0;
        DICE = 0;
        IOU = 0;
        REC = 0;
        SPE = 0;
        
    else
        ACC = (TP + TN)/(TP + TN + FN + FP);
        DICE = (2 * TP) / (2 * TP + FP + FN);
        IOU = TP/(TP + FP + FN);
        REC = TP/(TP + FN);
        SPE=TN/(TN+FP);
        
      
    end

    
    x = [ACC, DICE,IOU,REC,SPE];
    fprintf('tp:%d, fn:%d, fp:%d, tn:%d\n',TP,FN,FP,TN)
    fprintf('%f, %f, %f, %f,%f\n',ACC, DICE,IOU,REC,SPE)
    disp('----------------------------------------------------------------------------')
end