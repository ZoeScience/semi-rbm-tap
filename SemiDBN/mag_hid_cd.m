function cd=mag_hid_cd(rbm,opts,m_vis,m_hid)
    %buf = repmat(rbm.c', size(m_vis,1), 1) + m_vis * rbm.W';
    buf = bsxfun(@plus,rbm.c',m_vis * rbm.W');
    second_order = ( (m_vis - m_vis.^2) * rbm.W2'  ) .* (0.5 - m_hid);
    buf = buf + second_order;
    cd = sigm(buf);
end
