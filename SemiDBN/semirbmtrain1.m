function rbm = semirbmtrain(rbm, x, opts,a)
    assert(isfloat(x), 'x must be a float');
    assert(all(x(:)>=0) && all(x(:)<=1), 'all data in x must be in [0:1]');
    m = size(x, 1);
    numbatches = m / opts.batchsize;
    
    assert(rem(numbatches, 1) == 0, 'numbatches not integer');
    [nh,nv]=size(rbm.W);
    
	ll_preTrain = mean(persudoLL(rbm,x))/(nh+nv);
	
	disp(['epoch 0/' num2str(opts.numepochs)  '. Persudo Likelihoodis: ' num2str(ll_preTrain)]);
	
    for i = 1 : opts.numepochs
        kk = randperm(m);
        err = 0;
        ll = 0;
%        scoreTAP=0;
        recon_error=0;
        recon_error_1=0;
        recon_error_2=0;
        
%         if i >= 2       %第3次以后扔掉显层中低于阈值的连接
%             threshold =0;
%             if(threshold >= 0)
%                     C_threshold = (rbm.C = threshold);
%                     rbm.C = rbm.C .* C_threshold;
%                     C2_threshold = (rbm.C2 = threshold);
%                     rbm.C2 = rbm.C2 .* C2_threshold;
%             end
%         end

%          if i>=2   %第n次以后扔掉显层
%             rbm.C=zeros(size(rbm.C));
%             rbm.C2=zeros(size(rbm.C2));
%          end 

%%%%%%%%inverse ising问题需要扔掉隐层和显隐之间的权重
%            if i >100
%                rbm.W=zeros(size(rbm.W));
%                rbm.W2=zeros(size(rbm.W2)); 
%                rbm.c=zeros(size(rbm.c));   % rbm.c是隐层的偏执
%                
%            end 

% less than threshold, dropout W
%         if i >= 2
%             threshold =0;
%             if(threshold > 0)
%                     W_threshold = (abs(rbm.W) > threshold);
%                     rbm.W = rbm.W .* W_threshold;
%                     W2_threshold = (rbm.W2 > threshold);
%                     rbm.W2 = rbm.W2 .* W2_threshold;
%             end
%               
%             W_Zeros=find(W_threshold == 0);
%             a=length(W_Zeros);
%             disp(['A:' num2str(a)]);
%             rbm.C=zeros(size(rbm.C));
%             rbm.C2=zeros(size(rbm.C));
%         end
        
         
        for l = 1 : numbatches
            batch = x(kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize), :);
            
            %if opts.approx == 'CD'
            v_pos = batch;
            h_pos=sigm(repmat(rbm.c', opts.batchsize, 1) + v_pos * rbm.W');%for 
            
            %h_sample = sigmrnd(repmat(rbm.c', opts.batchsize, 1) + v_pos * rbm.W');%for MCMC Sampling
            h_samples = binornd(1,h_pos,[size(h_pos)]);

             if strcmp(opts.approx,'CD')
                     vis_init = v_pos;
                     hid_init = h_samples;
                     [v_neg, h_samples1, v_means1, h_neg] = MCMC(rbm, opts,vis_init,hid_init,'hidden');
             elseif strcmp(opts.approx,'tap2') ||  strcmp(opts.approx,'tap3')   ||  strcmp(opts.approx,'naive') 
                     vis_init = v_pos;
                     hid_init = h_pos;
                     [v_neg, h_neg] = equilibrate(rbm,opts,vis_init,hid_init);
             end
            
             
            %if i == 2
            rbm=calculate_weight_gradient(rbm,i,opts,v_pos,h_samples,v_neg,h_neg);            
            rbm=calculate_weight_gradient_c(rbm,i,opts,v_pos,h_samples,v_neg,h_neg);            

            rbm.vb = rbm.momentum * rbm.vb + rbm.alpha * sum(v_pos - v_neg)' / opts.batchsize;
            rbm.vc = rbm.momentum * rbm.vc + rbm.alpha * sum(h_samples - h_neg)' / opts.batchsize; 
           
%              else
%                 rbm.vc=zeros(size(rbm.vc));
%                 
%             end 
            
             
            
            if strcmp(opts.weight_decay,'l2')
                   rbm.vW = rbm.vW - rbm.alpha * opts.regularize * rbm.W;
                 %  rbm.vC = rbm.vC - rbm.alpha * opts.regularize * rbm.C;
            end
            if strcmp(opts.weight_decay,'l1')
                   rbm.vW = rbm.vW - rbm.alpha * opts.regularize * sign(rbm.W);
                 %  rbm.vC = rbm.vC - rbm.alpha * opts.regularize * sign(rbm.C);
            end 
            
            
            rbm.W = rbm.W + rbm.vW;  
            %if strcmp(opts.approx,'semi')
            rbm.C = rbm.C + rbm.vC;
            rbm.C2 = rbm.C.^2;
            rbm.b = rbm.b + rbm.vb;
            rbm.c = rbm.c + rbm.vc;
            
            %end
         if strcmp(opts.approx,'tap2') || strcmp(opts.approx,'tap3')
                rbm.W2 = rbm.W.^2;
         end
         if strcmp(opts.approx,'tap3')
                 rbm.W3 = rbm.W.^3;
         end
                                              
         err = err + sum(sum((v_pos - v_neg) .^ 2)) / opts.batchsize;
         ll = ll + mean(persudoLL(rbm,v_pos))/(nh+nv);
%         scoreTAP=scoreTAP+mean(score_samples_TAP(rbm,opts,v_pos))/(nh+nv);
         recon_error=reconstruction_error(rbm,v_pos)/(nh+nv);
         diff_abs = sum(sum(abs(v_pos-v_neg)));
         
         diff_abs1 = sum(sum((v_pos-v_neg).^2));
  
         
         
        end   %batch
        
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)  '. Average reconstruction error is: ' num2str(err / numbatches)]);
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)  '. Persudo Likelihoodis: ' num2str(ll / numbatches)]);
 %       disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)  '. TAP Score: ' num2str(scoreTAP / numbatches)]);
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)  '. Recon Error: ' num2str(recon_error / numbatches)]);

    end   %epoch
end

function rbm=calculate_weight_gradient(rbm,i,opts,v_pos,h_pos,v_neg,h_neg) %gradient descent of W (between two layers)
      c1 = h_pos' * v_pos;%if use h_pos instead of h_sample , the reconstruction error will swill
      c2 = h_neg' * v_neg;
      vW = c1-c2;
      
      %change1 对应dropout W
%       if i >= 2
%         threshold =0.001;
%         W_threshold = (abs(rbm.W) > threshold);
%       end
      
      if strcmp(opts.approx,'tap2') || strcmp(opts.approx,'tap3') %
           %disp('tap2')
           
           
           % buf2 = (h_neg-abs(h_neg).^2)' * (v_neg-abs(v_neg).^2) .* rbm.W; % T=1
            buf2 = (1-abs(h_neg).^2)' * (1-abs(v_neg).^2) .* rbm.W; % -1,1
           
          if  i<=100
             % vW =  vW - 10*  buf2; % T=0.1
            vW = vW -rbm.alpha * buf2;  %  T=1
           %  vW = vW - 0.5 *  buf2;  % T=2;
           % vW = vW - 0.2 *  buf2; % T=5
           % vW = vW - 0.1* buf2; % T=10
           % vW = vW - 2* buf2; % T=0.5
          else
              vW=zeros(size(vW));
          end 
        
      end
      if strcmp(opts.approx,'tap3')
            %disp('tap3')
            buf3 = ((h_neg-abs(h_neg).^2) .* (0.5-h_neg))' * (v_neg - abs(v_neg).^2) .* rbm.W2;%100*784
            vW = vW - 2.0 *rbm.alpha* buf3;
      end
      
         rbm.vW = rbm.vW * opts.momentum + opts.alpha * vW / opts.batchsize;
      
      %change1 对应dropout W
%       if i >= 2
%         rbm.vW = rbm.vW .* W_threshold;
%       end
      rbm.vW_prev = rbm.vW;
end

function rbm=calculate_weight_gradient_c(rbm,i,opts,v_pos,h_pos,v_neg,h_neg) % gradient of C(visible layer)
    vC = (v_pos' * v_pos) - (v_neg' * v_neg) ;
    
%      if i >= 2   %对应着扔掉C的权重更新
%          threshold =0;
%          C_threshold = (abs(rbm.C) ~= threshold);
%     end
    
    if strcmp(opts.approx,'tap2') 
        
        % buf=(v_pos - abs(v_pos).^2)' * (v_neg - abs(v_neg).^2) .* rbm.vC; %tap-plefka T=1
         buf=(1 - abs(v_pos).^2)' * (1 - abs(v_neg).^2) .* rbm.vC; %-1,1 
         
        vC = vC -rbm.alpha * buf; % T=1
       %  vC = vC - 0.5 * buf; % T=2 
       %  vC = vC - 0.2 * buf; % T=5 
       %  vC = vC - 0.1 * buf; % T=10
       % vC = vC - 2* buf; % T=0.5 
       % vC = vC - 10* buf; % T=0.1
       
    end
    rbm.vC = rbm.vC * opts.momentum  + vC * opts.alpha / opts.batchsize;
    
%     if i >= 2   %对应着扔掉C
%        rbm.vC = rbm.vC .* C_threshold;
%     end 
end

function energy = free_energy(rbm,vis)
    %size(vis(1,:))
    %size(rbm.b)%784*1
    for i=1:size(vis,1)
        vis_bias(i,:) = vis(i,:) .* rbm.b';
    end
    
    vb = sum(vis_bias,2);
    %size(vb):batchsize * 1
    hid_bias = vis * rbm.W' ;
    for i=1:size(hid_bias,1)
        hid_bias(i,:) = hid_bias(i,:) + rbm.c';
    end
    Wx_b_log = sum(log(1+exp(hid_bias)),2);
    %size(Wx_b_log):batchsize * 1
    energy=-vb-Wx_b_log;
end

function LL=persudoLL(rbm,vis)
    [n_samples,n_feat] = size(vis);
    vis_corrupted=vis;
    %idxs=round(rand(1,n_samples)*n_feat)+1;
    %idxs=randint(1,n_samples,[1 n_feat]);
    idxs=randint_s(1,n_samples,1,n_feat);
    for i=1:n_samples
        %for i =1:idxs(j)
        j=idxs(i);
        vis_corrupted(i,j) = 1 - vis_corrupted(i,j);
    end
    fe=free_energy(rbm,vis);
    fe_corrupted=free_energy(rbm,vis_corrupted);
    LL=n_feat * logsigm(fe_corrupted-fe);
end

function X = logsigm(P)
    X = log(1./(1+exp(-P)));
end
    
function mse = reconstruction_error(rbm,vis)
    hid=sigmrnd(repmat(rbm.c', size(vis,1), 1) + vis * rbm.W');
    vis_rec = sigmrnd(repmat(rbm.b', size(vis,1), 1) + hid* rbm.W);
    dif = vis_rec - vis;
    mse = mean(mean(dif .* dif));
end

% function score = score_samples_TAP(rbm,opts,vis)
    % [v_pos,h_pos,m_vis,m_hid] = iter_mag(rbm,opts,vis);
    % eps=1e-6;
    % m_vis = max(m_vis, eps);
    % m_vis = min(m_vis, 1.0-eps);
    % m_hid = max(m_hid, eps);
    % m_hid = min(m_hid, 1.0-eps);

    % m_vis2 = m_vis.^2;
    % m_hid2 = m_hid.^2;
    
    % S= sum(m_vis .* log(m_vis) + (1.0 - m_vis) .* log(1.0 - m_vis),2) - sum(m_hid .* log(m_hid) + (1.0 - m_hid) .* log(1.0 - m_hid),2);
    % U_naive= - m_vis * rbm.b - m_hid * rbm.c - sum((m_vis * rbm.W') .* m_hid ,2);
    % Onsager= - 0.5 * sum(((m_vis-m_vis2) * rbm.W2') .* (m_hid - m_hid2),2);
    % fe_tap=U_naive + Onsager - S;
    % fe= free_energy(rbm,vis);
    % score=fe_tap-fe;
% end

% function [v_pos, h_pos, m_vis, m_hid]=iter_mag(rbm,opts,vis)
    % v_pos = vis;
    % h_pos = sigmrnd(repmat(rbm.c', size(vis,1), 1) + v_pos * rbm.W');
    
    % if strcmp(opts.approx,'CD')
        % mag_vis = @mag_vis_cd;
        % mag_hid = @mag_hid_cd;
    % elseif strcmp(opts.approx,'tap2')
        % mag_vis = @mag_vis_tap2;
        % mag_hid = @mag_hid_tap2;
    % end
    % m_vis = 0.5 * mag_vis(rbm, opts,vis, h_pos) + 0.5 * vis;
    % m_hid = 0.5 * mag_hid(rbm, opts,m_vis, h_pos) + 0.5 * h_pos;
    % for i = 1:1:opts.iterations-1
        % m_vis = 0.5 * mag_vis(rbm, opts,m_vis, m_hid) + 0.5 * m_vis;
        % m_hid = 0.5 * mag_hid(rbm, opts,m_vis, m_hid) + 0.5 * m_hid;
    % end
% end

function [vis_samples, hid_samples, vis_means, hid_means]=MCMC(rbm, opts,v_init,h_init,StartMode)
    iterations=opts.iterations;
    if strcmp(StartMode,'hidden')
        hid_samples=h_init;
        hid_means=h_init;
        iterations = iterations + 1;
    end
    vis_samples=v_init;
    vis_means=v_init;
    for i = 1:iterations-1
        [vis_samples,vis_means] = sample_visibles(rbm,opts,vis_samples,hid_samples);
        [hid_samples, hid_means] = sample_hiddens(rbm,opts,vis_samples);
    end
end

function [samples,means]=sample_visibles(rbm,opts,m_vis,m_hid)
    %rows = size(m_vis,2);
    m_vis_buf1 = zeros(size(m_vis));
    C_tmp = rbm.C;
    % for i=1:rows
        % C_tmp(i,i)=0;
        % C_tmp2(i,i)=0;
    % end
    
	C_tmp=C_tmp-diag(diag(C_tmp));
	
    m_vis_buf1 = m_vis * C_tmp;
    
    %buf = repmat(rbm.b', size(m_hid,1), 1) + m_hid * rbm.W ;
    % buf = bsxfun(@plus,rbm.b',m_hid * rbm.W);
    % second_order = ( (m_hid - m_hid.^2 ) * rbm.W2 ) .* (0.5 - m_vis);
    
	means=sigm(bsxfun(@plus,rbm.b',m_hid*rbm.W) + m_vis_buf1);
    samples=binornd(1,means,size(means));
	
	
    % buf = buf + second_order;
    % buf = buf + m_vis_buf1;
    % buf = buf + m_vis_buf2;
    % cd_mean = sigm(buf);
    % %cd=normrnd(cd_mean,1,[size(buf)]);
    % cd = sigmrnd(buf);
end

function [samples,means]=sample_hiddens(rbm,opts,m_vis)
    %buf = repmat(rbm.c', size(m_vis,1), 1) + m_vis * rbm.W';
	
	means=sigm(bsxfun(@plus,rbm.c',m_vis * rbm.W'));
    samples=binornd(1,means,size(means));
	
	
    % buf = bsxfun(@plus,rbm.c',m_vis * rbm.W');
    % second_order = ( (m_vis - m_vis.^2) * rbm.W2'  ) .* (0.5 - m_hid);
    % buf = buf + second_order;
    % cd_mean = sigm(buf);
    % cd = sigmrnd(buf);		
    %cd=binornd(1,cd_mean,size(cd_mean));
end

function [m_vis,m_hid] = equilibrate(rbm,opts,m_vis,m_hid)
    if strcmp(opts.approx,'tap3')
        mag_vis = @mag_vis_tap3;
        mag_hid = @mag_hid_tap3;
    elseif strcmp(opts.approx,'tap2')
        mag_vis = @mag_vis_tap2;
        mag_hid = @mag_hid_tap2;
    end
    for i = 1:opts.iterations
        m_vis = 0.5 * mag_vis(rbm, opts,m_vis, m_hid) + 0.5 * m_vis;
        m_hid = 0.5 * mag_hid(rbm, opts,m_vis, m_hid) + 0.5 * m_hid;
    end
end

% function tap2 = mag_vis_tap2(rbm,opts,m_vis,m_hid)
%     rows = size(m_vis,2);
%     %disp('Vis');
%     buf1=sigm(bsxfun(@plus,rbm.b',m_hid * rbm.W));    
%     C_tmp = zeros(size(rbm.C));
%     C2_tmp = zeros(size(rbm.C));
% %     
%     %take only one neighboring pixe
%     C_tmp(1,2)=rbm.C(1,2);
%     C2_tmp(1,2)=rbm.C2(1,2);
%     for i=2:rows-1
%         C_tmp(i,i-1)=rbm.C(i,i-1);
%         C_tmp(i,i+1)=rbm.C(i,i+1);
%         C2_tmp(i,i-1)=rbm.C2(i,i-1);
%         C2_tmp(i,i+1)=rbm.C2(i,i+1);
%     end
%     C_tmp(rows,rows-1)=rbm.C(rows,rows-1);
%     C2_tmp(rows,rows-1)=rbm.C2(rows,rows-1);
%     
% %  %   take two neighboring pixes
% %     for i=1:rows
% %         if (i-1) > 0
% %             C_tmp(i,i-1)=rbm.C(i,i-1);
% %             C2_tmp(i,i-1)=rbm.C2(i,i-1);
% %         end
% %         if (i-2) > 0
% %             C_tmp(i,i-2)=rbm.C(i,i-2);
% %             C2_tmp(i,i-2)=rbm.C2(i,i-2);
% %         end
% %         if (i+1) <= rows
% %             C_tmp(i,i+1)=rbm.C(i,i+1);
% %             C2_tmp(i,i+1)=rbm.C2(i,i+1);
% %         end
% %         if (i+2) <= rows
% %             C_tmp(i,i+2)=rbm.C(i,i+2);
% %             C2_tmp(i,i+2)=rbm.C2(i,i+2);
% %         end
% %     end
% 
%     buf2=m_vis * C_tmp;
%     buf3=(m_hid - m_hid.^2) * rbm.W2 .* (0.5-m_vis);
%     buf4=(m_vis - m_vis.^2) * C2_tmp .* (0.5-m_vis);
%     tap2 = sigm(buf1 + buf2 + buf3 + buf4);
% end

% %full connected visible layer 
function tap2 = mag_vis_tap2(rbm,opts,m_vis,m_hid)
    [rows,cols] = size(m_vis);
    %rows = size(m_vis,2);
    buf1=bsxfun(@plus,rbm.b',m_hid * rbm.W);
    C_tmp = rbm.C;
    C_tmp2 = rbm.C2;
    %for i = 1:cols
    for i=1:rows
        C_tmp(i,i)=0;
        C_tmp2(i,i)=0;
    end
  
    buf2=m_vis * C_tmp; %T=1
    
     buf3=(m_hid- m_hid.^2) * rbm.W2 .*(0.5- m_vis);   % 0,1
     buf4=(m_vis - m_vis.^2) * C_tmp2 .* (0.5-m_vis);  % 0,1 
    
     tap2 = sigm(buf1 + buf2 + buf3 + buf4); %T=1;
    % tap2 = sigm( buf1 + buf2 +0.5* buf3 +0.5* buf4); % T=2;
    % tap2 = sigm(buf1 + buf2 + 0.2* buf3 + 0.2* buf4); % T=5
    % tap2 = sigm(buf1 + buf2 +0.1 * buf3 +0.1* buf4); % T=10
    % tap2 = sigm( buf1 + buf2 +2* buf3 +2* buf4); % T=0.5;
    % tap2 = sigm( buf1 + buf2 + 10* buf3 + 10* buf4); % T=0.1
end

%Random Connected Visible Layer
% function tap2 = mag_vis_tap2(rbm,opts,m_vis,m_hid) %visible layer with stochastic matrix
%     [rows,cols] = size(m_vis);
%     %disp('Vis');
%     buf1=bsxfun(@plus,rbm.b',m_hid * rbm.W);
%     C_tmp = zeros(size(rbm.C));
%     C_tmp2 = zeros(size(rbm.C));
%     dropoutFraction =0.5;
%       if(dropoutFraction > 0)
%                 C_dropOutMask = (rand(size(C_tmp)) > dropoutFraction);
%                 C_tmp = rbm.C .* C_dropOutMask;
%                 C2_dropOutMask = (rand(size(C_tmp2)) > dropoutFraction);
%                 C_tmp2 = rbm.C2 .* C2_dropOutMask;
%        end
% 
%     buf2=m_vis * C_tmp;
% 
% %     buf3=(m_hid - m_hid.^2) * rbm.W2 .* (0.5-m_vis); % plefka
% %     buf4=(m_vis - m_vis.^2) * C_tmp2 .* (0.5-m_vis); % plefka
% 
%        buf3=(1- m_hid.^2) * rbm.W2 .*(- m_vis); % -1,1
%        buf4=(1 - m_vis.^2) * C_tmp2 .* (- m_vis); %-1,1
% 
%  %    buf3=(1- m_hid.^2) * rbm.W2 .*( m_vis); % -1,1 T=1
%  %    buf4=(1 - m_vis.^2) * C_tmp2 .* ( m_vis); %-1,1 T=1
%      
% %    tap2 = sigm(buf1 + buf2 + buf3 + buf4); % T=1
% %     tap2 = sigm( buf1 + buf2 +0.5* buf3 +0.5* buf4); % T=2;
% %     tap2 = sigm(buf1 + buf2 + 0.2* buf3 + 0.2* buf4); % T=5
% %     tap2 = sigm(buf1 + buf2 +0.1 * buf3 +0.1* buf4); % T=10
% %     tap2 = sigm( buf1 + buf2 +2* buf3 +2* buf4); % T=0.5;
%      tap2 = sigm( buf1 + buf2 + 10* buf3 + 10* buf4); % T=0.1
% end

%GPU Version for Dropout connection
% function tap2 = mag_vis_tap2(rbm,opts,m_vis,m_hid) %visible layer with stochastic matrix
%     [rows,cols] = size(m_vis);
%     %disp('Vis');
%     buf1=bsxfun(@plus,rbm.b',m_hid * rbm.W);
%     C_tmp = zeros(size(rbm.C));
%     C2_tmp = zeros(size(rbm.C));
%     GC_tmp=gpuArray(C_tmp);
%     GC2_tmp=gpuArray(C2_tmp);
%     GC=gpuArray(rbm.C);
%     dropoutFraction =0.8;
%       if(dropoutFraction > 0)
%                 GC_dropOutMask = rand(size(C_tmp),'gpuArray')>dropoutFraction;
%                 GC_tmp = rbm.C .* C_dropOutMask;
%                 C2_dropOutMask = (rand(size(C_tmp2))>dropoutFraction);
%                 C_tmp2 = rbm.C2 .* C2_dropOutMask;
%         end
%     buf2=m_vis * C_tmp;
%     buf3=(m_hid - m_hid.^2) * rbm.W2 .* (0.5-m_vis);
%     buf4=(m_vis - m_vis.^2) * C_tmp2 .* (0.5-m_vis);
%     tap2 = sigm(buf1 + buf2 + buf3 + buf4);
% end


function tap2 = mag_hid_tap2(rbm,opts,m_vis,m_hid)
    %disp('Hid');
    buf=bsxfun(@plus,rbm.c',m_vis * rbm.W');
   
    second_order=((m_vis - m_vis.^2) * rbm.W2' .* (0.5 - m_hid)); % 0,1 
  
 %  second_order=((1 - m_vis.^2) * rbm.W2' .* ( - m_hid));%-1,1,
 %  second_order=((1 - m_vis.^2) * rbm.W2' .* (  m_hid)); % -1,1,T=1
    
    buf = buf + second_order; % T=1 
   % buf =  buf + 0.5* second_order; % T=2 
   % buf = buf + 0.2* second_order; % T=5
   % buf = buf + 0.1 * second_order; % T=10
   % buf =  buf + 2* second_order; % T=0.5
   % buf = buf + 10* second_order; % T=0.1
    
    tap2=sigm(buf); % 公用
   
end