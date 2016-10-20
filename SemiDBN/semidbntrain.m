function dbn = semidbntrain(dbn, x, opts)
    n = numel(dbn.rbm);
    dbn.rbm{1} = semirbmtrain(dbn.rbm{1}, x, opts);
    for i = 2 : n
        if strcmp(opts.approx,'CD')
            [h_samples,h_init]=samples_hidden(dbn.rbm{i-1},opts,x);
            x=mag_hid_cd(dbn.rbm{i-1},opts,x,h_init);
        %x = semirbmup(dbn.rbm{i - 1}, x);
        elseif strcmp(opts.approx,'tap2')
            [h_samples,h_init]=samples_hidden(dbn.rbm{i-1},opts,x);
            m_vis = 0.5 * mag_vis_tap2(dbn.rbm{i-1}, opts,x, h_init) + 0.5 * x;
            x = 0.5 * mag_hid_tap2(dbn.rbm{i-1}, opts,m_vis, h_init) + 0.5 * h_init;
            %x=mag_hid_tap2(dbn.rbm{i-1},opts,x,h_init);
        end
        dbn.rbm{i} = semirbmtrain(dbn.rbm{i}, x, opts);
    end
end

function [h_samples,h_means]=samples_hidden(rbm,opts,vis)
        v_pos = vis;
        h_means=sigm(repmat(rbm.c', size(vis,1), 1) + v_pos * rbm.W');%for EMFSampling
            %h_sample = sigmrnd(repmat(rbm.c', opts.batchsize, 1) + v_pos * rbm.W');%for MCMC Sampling
        h_samples = binornd(1,h_means,[size(h_means)]);
end

function tap2 = mag_vis_tap2(rbm,opts,m_vis,m_hid)
    rows = size(m_vis,2);
    %disp('Vis');
    buf1=bsxfun(@plus,rbm.b',m_hid * rbm.W);    
    C_tmp = zeros(size(rbm.C));
    C2_tmp = zeros(size(rbm.C));
    
%     
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
    
 %   take two neighboring pixes
%     for i=1:rows
%         if (i-1) > 0
%             C_tmp(i,i-1)=rbm.C(i,i-1);
%             C2_tmp(i,i-1)=rbm.C2(i,i-1);
%         end
%         if (i-2) > 0
%             C_tmp(i,i-2)=rbm.C(i,i-2);
%             C2_tmp(i,i-2)=rbm.C2(i,i-2);
%         end
%         if (i+1) <= rows
%             C_tmp(i,i+1)=rbm.C(i,i+1);
%             C2_tmp(i,i+1)=rbm.C2(i,i+1);
%         end
%         if (i+2) <= rows
%             C_tmp(i,i+2)=rbm.C(i,i+2);
%             C2_tmp(i,i+2)=rbm.C2(i,i+2);
%         end
%     end
%     
    
    
    buf2=m_vis * C_tmp;
    buf3=(m_hid - m_hid.^2) * rbm.W2 .* (0.5-m_vis);
    buf4=(m_vis - m_vis.^2) * C2_tmp .* (0.5-m_vis);
    tap2 = sigm(buf1 + buf2 + buf3 + buf4);
end