function tap2 = mag_hid_tap2(rbm,opts,m_vis,m_hid)
    buf=bsxfun(@plus,rbm.c',m_vis * rbm.W');
    second_order=((m_vis - m_vis.^2) * rbm.W2' .* (0.5 - m_hid));
    buf = buf + second_order;
    tap2=sigm(buf);
end

