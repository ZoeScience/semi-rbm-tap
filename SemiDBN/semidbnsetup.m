function dbn = semidbnsetup(dbn, x, opts)
    n = size(x, 2);
    disp('Setup')
    
    dbn.sizes = [n, dbn.sizes]; %dbn.sizes初值[100,100]，执行结果是[784,100,100]
    dbn.sizes;
    for u = 1 : numel(dbn.sizes) - 1 % numel()返回元素个数
        dbn.rbm{u}.alpha    = opts.alpha;
        dbn.rbm{u}.momentum = opts.momentum;

        %dbn.rbm{u}.W  = randn(dbn.sizes(u + 1), dbn.sizes(u));
        dbn.rbm{u}.W  = zeros(dbn.sizes(u + 1), dbn.sizes(u));
%         disp('size W')
%         size(dbn.rbm{u}.W)
        dbn.rbm{u}.W2 = dbn.rbm{u}.W.^2;
%         disp('size W2')
%         size(dbn.rbm{u}.W2)
        dbn.rbm{u}.W3 = dbn.rbm{u}.W.^3 ;
        
        dbn.rbm{u}.C = zeros(dbn.sizes(u),dbn.sizes(u));
         disp('size C')
         size(dbn.rbm{u}.C)
        dbn.rbm{u}.C2 = dbn.rbm{u}.C.^2;
        %dbn.rbm{u}.C3 = dbn.rbm{u}.C.^3;
        
        dbn.rbm{u}.vC = zeros(dbn.sizes(u),dbn.sizes(u));
        dbn.rbm{u}.vC_prev = zeros(dbn.sizes(u),dbn.sizes(u));
        
        dbn.rbm{u}.vW = zeros(dbn.sizes(u + 1), dbn.sizes(u));
        dbn.rbm{u}.vW_prev = zeros(dbn.sizes(u + 1), dbn.sizes(u));

        dbn.rbm{u}.b  = zeros(dbn.sizes(u), 1); %b是显层
        dbn.rbm{u}.vb = zeros(dbn.sizes(u), 1);

        dbn.rbm{u}.c  = zeros(dbn.sizes(u + 1), 1); %c是隐层
        dbn.rbm{u}.vc = zeros(dbn.sizes(u + 1), 1);

%         dbn.rbm{u}.vC = randn(dbn.sizes(u),dbn.sizes(u));
%         dbn.rbm{u}.vC_prev = randn(dbn.sizes(u),dbn.sizes(u));
%         
%         dbn.rbm{u}.vW = randn(dbn.sizes(u + 1), dbn.sizes(u));
%         dbn.rbm{u}.vW_prev = randn(dbn.sizes(u + 1), dbn.sizes(u));
% 
%         dbn.rbm{u}.b  = randn(dbn.sizes(u), 1); %b是显层
%         dbn.rbm{u}.vb = randn(dbn.sizes(u), 1);
% 
%         dbn.rbm{u}.c  = randn(dbn.sizes(u + 1), 1); %c是隐层
%         dbn.rbm{u}.vc = randn(dbn.sizes(u + 1), 1);


    end
end