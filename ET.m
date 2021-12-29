clear all
close all
clc
for k=1:num
    A=load(['path',num2str(k),'.mat']);
    outdir='outpath';
    A0=A.predict;
    A0 = im2uint16(A0)/65535;
    B = permute(A0,[3,2,1]);
    
    bb = z_deal(B);
    bb1 = bb(3,:);
    [M,N] = size(bb1);
    
    
    for i = 1:length(bb1)
        
    
        
        aa = imfill(B(:,:,bb1(i)));
        
        se = strel('disk',1);ee0 = imopen(aa,se) * 65535;
        se = strel('disk',2);ee2 = imopen(aa,se) * 65535;
        ee3 = uint16((ee0 + ee2)/65535);
        figure,imshow(aa,[])
        
        if sum(ee3,'ALL')~=0
            ee1= z_zdltqy(ee3);    
            skeleton0 = bwmorph(ee1,'skel',inf);
            pp0 = find(skeleton0 == 1);cc = ee1;cc(pp0) = 2;
            
            if sum(skeleton0,'ALL')~=0
                figure();imshow(cc,[])
                dd = z_terminal(skeleton0);
                if size(dd,1) == 1
                    bb(4,i) = 0;
                else
                    if size(dd,1) == 2
                        [U,V] = find(skeleton0 == 1);
                        za2 = [U,V];
                    else
                        dd1 = dd;
                        skeleton1 = skeleton0;
                        while(size(dd1,1) > 2)
                            Q = z_branch(dd1,skeleton1);
                            za = z_gu_branch(Q);
                            za1 = z_gu_chongfu(za);
                            za2 = z_zx(za1,skeleton1);
                            skeleton1 = z_deal2(ee1,za2);
                            figure;imshow(skeleton1,[])
                            skeleton1 = z_zdltqy(skeleton1);
                            dd1 = z_terminal(skeleton1);
                        end
                    end
                    
                    za3 = z_zx_tj3(za2);
                    if size(za3,1)>1
                        za4 = z_zx_bc(za3,ee1);
                        figure( 'Name', strcat('NO.',num2str(i),' ,Pic.',num2str(bb1(i))) );
                        bb(4,i) = z_hd_cal(za4,ee1);
                    else
                        bb(4,i) = 0;
                    end
                end
            end
        else
            bb(4,i) = 0;
        end
    end

    savepath=strcat(outdir,num2str(k),'.mat');
    save(savepath,'bb');
    
end

function juli = z_hd_cal(za4,ee1)  

    za5 = za4';
    za6 = flip(za5);
    za7 = za6';
    za7(:,2) = za7(:,2) + 32;
    
    ee2 = zeros(32*3,224);
    ee2(33:64,:) = ee1;
    se2=[0 1 0;1 1 1;0 1 0];
    
    ee3=imerode(ee2,se2);    
    ee4 = ee2 + ee3;
    [U,V] = find(ee4 == 1);
    bz_4(:,1) = V;bz_4(:,2) = U;

    imshow(ee2,[]);
    for kk = 1:length(za7)
        hold on;
        plot(za7(kk,1),za7(kk,2),'b.');
    end
      
    for kk = 1:length(bz_4)
        hold on;
        plot(bz_4(kk,1),bz_4(kk,2),'r.');
    end
    
    e = [];
    if size(za7,1)<3
        e=0;
    else
        for i = 2:size(za7,1)-1
            
            a = za7(i-1,:);
            b = za7(i,:);
            c = za7(i+1,:);
            
            if (a(1) == b(1) && a(1) ~= c(1)) || (a(1) == b(1) && a(1) == c(1) && b(1) == c(1))
                d = 0;
            elseif a(2) == b(2) ||  b(2) == c(2)
                d1 = find(bz_4(:,1) == b(1));%
                if length(d1) == 1
                    d = 0;
                else
                    d = max(abs(bz_4(d1(1),2) - bz_4(d1(2),2)));
                end
            elseif ((c(2)-b(2))/(c(1)-b(1))) ==  ((b(2)-a(2))/(b(1)-a(1)))
                m1 = (c(2)-b(2))/(c(1)-b(1));
                m = -1/m1;
                n = b(2) - m*b(1);
                d0 = [];
                d1 = 1;
                for j = 1:length(V)
                    x0 = bz_4(j,1);
                    y0 = bz_4(j,2);
                    d0(d1) = abs(m*x0-y0+n)/sqrt((m*m)+1);
                    d1 = d1 +1;
                end
                
                bz_2 = bz_4;
                [~,d3] = min(d0);
                po1 = bz_4(d3,:);
                d0(d3) = [];
                
                if po1(2)<b(2)
                    flag = 1;
                    while flag ~= 3
                        [~,d3] = min(d0);
                        if bz_2(d3,2)>b(2)
                            po2 = bz_2(d3,:);
                            flag = 3;
                        else
                            d0(d3) = [];bz_2(d3,:)=[];
                        end
                    end
                else
                    flag = 2;
                    while flag ~= 3
                        [~,d3] = min(d0);
                        if bz_2(d3,2)<b(2)
                            po2 = bz_2(d3,:);
                            flag = 3;
                        else
                            d0(d3) = [];bz_2(d3,:)=[];
                        end
                    end
                end
                d = sqrt((po1(1)-po2(1))^2 + (po1(2)-po2(2))^2);
            else
                bz_3 = bz_4;
                [center, ~] = w_circle(za7(i-1,:), za7(i,:), za7(i+1,:));
                [m, n] = z_line(b, center);
                d0 = [];
                d1 = 1;
                for j = 1:length(V)
                    x0 = bz_4(j,1);
                    y0 = bz_4(j,2);
                    d0(d1) = abs(m*x0-y0+n)/sqrt((m*m)+1);
                    d1 = d1 +1;
                end
                
                [~,d3] = min(d0);
                po1 = bz_4(d3,:);
                d0(d3) = [];
                
                if po1(2)<=b(2)
                    flag = 1;
                    while flag ~= 3
                        [~,d3] = min(d0);
                        if bz_3(d3,2)>b(2)
                            po2 = bz_3(d3,:);
                            flag = 3;
                        else
                            d0(d3) = []; bz_3(d3,:) = [];
                        end
                    end
                else
                    flag = 2;
                    while flag ~= 3
                        [~,d3] = min(d0);d2 = d3;
                        if bz_3(d3,2)<=b(2)
                            po2 = bz_3(d3,:);
                            flag = 3;
                        else
                            d0(d3) = [];bz_3(d3,:) = [];
                        end
                    end
                end
                d = sqrt((po1(1)-po2(1))^2 + (po1(2)-po2(2))^2);
            end
            e(i) = d;
        end
    end
    juli = max(e); 
end



function [k, b] = z_line(pt1, pt2)
    k = (pt2(2) - pt1(2))/(pt2(1) - pt1(1));
    b = pt1(2) - k * pt1(1);
end


function [center, r] = w_circle(pt1, pt2, pt3)

A = zeros(2, 2); B = zeros(2, 1);
[A(1, :), B(1)] = circle2line(pt1, pt2);
[A(2, :), B(2)] = circle2line(pt2, pt3);
center = A\B;
r = norm(pt1' - center);
end

function [A, B] = circle2line(pt1, pt2)

A = 2*(pt1 - pt2);
B = norm(pt1)^2 - norm(pt2)^2;
end

function za6 = z_zx_bc(za3,ee1)  
    za4 = za3;
    for i = 2:size(za3,1)
        p(i) = abs(za3(i,1)-za3(i-1,1));
    end
    u = find(p>1);
    if u
        for j = 1:length(u)
            u1 = u(j);
            m = min(za3(u1,1),za3(u1-1,1)):max(za3(u1,1),za3(u1-1,1));
            n = ones(size(m)) * za3(u1,2);
            za4(end+1:end + length(m),:) = [m',n'];
        end
    end
    [za5(:,2),q] = sort(za4(:,2));
    qq = za4(:,1);
    za5(:,1) = qq(q);
    za60 = unique(za5,'row','stable');


    ee0 = zeros(size(ee1));
    for mm = 1:size(za60,1)
        ee0(za60(mm,1),za60(mm,2)) = 1;
    end

    ee0 = ee0.*double(ee1);
    [PP,QQ] = find(ee0 == 1);
    za61(:,1) = PP;
    za61(:,2) = QQ;

    [za6(:,2),q1] = sort(za61(:,2));
    qq = za61(:,1);
    za6(:,1) = qq(q1);

    jj = unique(za6(:,2));
    for k = 1:length(jj)
        f1 = find(za6(:,2) == jj(k));
        if length(f1)>1
            if sum(za6(f1,1) - za6(f1(1)-1,1))<0
                za6(f1,1) = sort(za6(f1,1),'descend');
            else
                za6(f1,1) = sort(za6(f1,1));
            end
        end
    end
end


function za6 = z_zx_tj3(za4) 
    [za5(:,2),k] = sort(za4(:,2)); 
    p = za4(:,1);
    za5(:,1) = p(k);
    za6 = za5;
    m = 0;
    for i = 1:size(za5,1)
        u = find(za6(:,2)==za5(i,2));
        if length(u)>1
            q = za6(:,1);
            f = q(u);
            za6(u(2:end),:)=[];
            za6(i-m,1) = round(sum(f)/length(u));
            m = m + length(u)-1;
        end
    end
end

function I1 = z_deal2(ee1,za2)
    I1 = uint16(zeros(size(ee1)));
    for i = 1:size(za2,1)
        a1 = za2(i,:);
        I1(a1(1,1),a1(1,2)) = 1;
    end
end
            
function cc = z_deal(B)
bb = 0;
    cc = [];
    k = 0;
    for i = 1:224
        aa = B(:,:,i);
        bb = sum(aa,'ALL');
        if  bb~=0 
            k = k + 1;
            aa1 = bwlabel(aa);               
            stats = regionprops(aa1,'Area');  
            cc(1,k) = length(stats);        
            cc(2,k) = bb;
            cc(3,k) = i;
        end
    end
end

function I7 = z_zdltqy(I5)

    [I6 ,num]= bwlabel(I5);
    if max(num)~=1
        I6=z_deal3(I6);
    end
    stats = regionprops(I6,'Area');
    area = cat(1,stats.Area);
    index = find(area == max(area));
    I7 = uint16(ismember(I6,index));  
end


function Q = z_terminal(A)  
    A = double(A);
    B = zeros(size(A) + 2);
    B(2:size(B,1)-1,2:size(B,2)-1) = A;
    [U,V] = find(B == 1);
    k = 1;
    for i = 1:length(U)       
        a = U(i);b = V(i);       
        P = double([B(a-1,b-1) B(a-1,b) B(a-1,b+1);
            B(a,  b-1) B(a,  b) B(a,  b+1);
            B(a+1,b-1) B(a+1,b) B(a+1,b+1)]);
        c = sum(P,'ALL');
        if c<3
            Q(k,1) = a-1;
            Q(k,2) = b-1;
            k = k + 1;
        end
    end
end

function ZZ = z_branch(aa,A)

    A = double(A);
    B = zeros(size(A) + 2);
    B(2:size(B,1)-1,2:size(B,2)-1) = A;
    
    aa(:,1) = aa(:,1)+1;
    aa(:,2) = aa(:,2)+1;

    k = 1;    
    for i = 1:size(aa,1)

        a = aa(i,1);b = aa(i,2);        
        ZZ(k,i) = {i};k = k + 1;
        ZZ(k,i) = {aa(i,:)-1};
        
        C = B;
        C(a,b) = C(a,b) - 1;      
        [ZZ,k,C,flag] = z_judge(ZZ,k,i,C);
        
        while(~ flag)
            [ZZ,k,C,flag] = z_judge(ZZ,k,i,C);
        end 
        k = 1;
    end
end


function [ZZ,k,C,flag] = z_judge(ZZ,k,i,C)

    flag = 0;
    Z1= ZZ{k,i};
    a = Z1(1)+1;b = Z1(2)+1;

    P = double([C(a-1,b-1) C(a-1,b) C(a-1,b+1);
                C(a,  b-1) C(a,  b) C(a,  b+1);
                C(a+1,b-1) C(a+1,b) C(a+1,b+1)]);
    [U,V] = find(P == 1);
    
    [U1,V1] = z_switch(U,V);
    a = a + U1;b = b + V1;

    P = double([C(a-1,b-1) C(a-1,b) C(a-1,b+1);
                C(a,  b-1) C(a,  b) C(a,  b+1);
                C(a+1,b-1) C(a+1,b) C(a+1,b+1)]);
    if sum(P,'ALL') == 2
        C(a,b) = C(a,b) - 1;k = k+1;
        ZZ(k,i) = {[a,b]-1};
    else
        k = k+1;
        ZZ(k,i) = {[a,b]-1};
        flag = 1;
    end
    
end

function [a,b] = z_switch(U,V)

if U == 1
    if V == 1
        a = -1;b = -1;
    elseif V == 2
        a = -1;b = 0;
    elseif V == 3
        a = -1;b = 1;
    end
elseif U ==2
    if V == 1
        a = 0;b = -1;
    elseif V == 2
        a = 0;b = 0;
    elseif V == 3
        a = 0;b = 1;
    end
elseif U == 3
    if V == 1
        a = 1;b = -1;
    elseif V == 2
        a = 1;b = 0;
    elseif V == 3
        a = 1;b = 1;
    end
end
end


function za = z_gu_branch(Q)

    [U,V] = size(Q);
    for z1 = 1:V 
        k = 1;
        a1 = [];
        for z2 = 2:U
            if Q{z2,z1}
                a1(k,:) = Q{z2,z1};
                k = k + 1;
            end
        end
        za(z1) = struct('data',flip(a1));
    end
end

function za = z_gu_chongfu(za)  
    ba = za;
    for i = 1:length(ba)
         a1 = ba(i).data;
         k = 0;
         c = [];
         jj = setdiff([1:length(za)],i,'stable');

         for j  = jj
            a2 = ba(j).data;
            b = intersect(a1,a2,'rows','stable');
            if b
                c(k + 1:k + size(b,1),:) = b;
                k = k + size(b,1);
            end
         end   
         
         if c
             d = unique(c,'rows','stable');
             e = setdiff(a1,d,'rows','stable');
             za(i).data = [];za(i).data = e;
             a1 = [];a2 = [];b = [];d = [];e = [];
         end
    end
end

function za4 = z_zx(za1,skeleton0)  
    
    za3 = za1;
    [U,V] = find(skeleton0 == 1);
    za2 = [U,V];
    uu=0;
    for i = 1:length(za1)
        a1 = za1(i).data;        
        if (size(a1,1)<2) || (a1(1,2) == a1(2,2))
            za2 = setdiff(za2,a1,'rows','stable');            
            za3(i-uu)=[];
            uu = uu+1;
        end
    end
    za4 = z_zx_tj2(za2,za3);     
end

function za2 = z_zx_tj2(za2,za3)  
   for i = 1:length(za3)
        a1 = za3(i).data;                
        if i==1
            max_x = a1(end,2);i1 = 1;
            min_x = a1(end,2);i2 = 1;
        else
            p = a1(end,2);
            if (p>max_x)
                i1 = i;
            end
            if (p<min_x)
                i2 = i;
            end
        end       
   end

   for j = 1:length(za3)
        if (j~=i1) && (j~=i2)
            b1 = za3(j).data;  
            za2 = setdiff(za2,b1,'rows','stable');        
        end
   end
end

function dd = z_deal3(aa)
    I = uint16(zeros(size(aa)));
    [U,V] = find(aa==1);
    aa1 = [U,V];
    for i = 1:length(U)
        a = U(i);b = V(i);
        I(U(i),V(i)) = aa(a,b);    
    end
 dd=I;
end


