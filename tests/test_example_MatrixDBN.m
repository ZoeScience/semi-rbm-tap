function test_example_MatrixDBN
load mnist_uint8;

n_samples=size(train_x,1)
row=28;
col=size(train_x,2)/row;
train_x = double(reshape(train_x,n_samples,row,col)) / 255;
test_x  = double(reshape(test_x,n_samples,row,col))  / 255;
train_y = double(train_y);
test_y  = double(test_y);

%%  ex1 train a 100 hidden unit RBM and visualize its weights
rand('state',0)
matrixdbn.sizes = [15,15];
opts.numepochs =   20;
opts.batchsize = 100;
opts.momentum  =   0.5;
opts.alpha     =   0.005;
opts.approx = 'tap2'
opts.regularize=0.01
opts.weight_decay='l1'
opts.iterations=2
matrixdbn = matrixdbnsetup(matrixdbn, train_x, opts);
matrixdbn = matrixdbntrain(matrixdbn, train_x, opts);
figure; visualize(matrixdbn.rbm{1}.W');   %  Visualize the RBM weights

%%  ex2 train a 100-100 hidden unit DBN and use its weights to initialize a NN
% rand('state',0)
% %train dbn
% dbn.sizes = [15,15];
% opts.numepochs =   10;
% opts.batchsize = 100;
% opts.momentum  =   0.5;
% opts.alpha     =   0.005;
% opts.approx = 'semi'
% opts.regularize=0.01
% opts.weight_decay='l1'
% opts.iterations=1
% 
% % dbn.sizes = [100 100];
% % opts.numepochs =   1;
% % opts.batchsize = 100;
% % opts.momentum  =   0;
% % opts.alpha     =   1;
% % opts.approx = 'CD'
% dbn = dbnsetup(dbn, train_x, opts);
% dbn = dbntrain(dbn, train_x, opts);
% 
% %unfold dbn to nn
% nn = dbnunfoldtonn(dbn, 10);
% nn.activation_function = 'sigm';
% 
% %train nn
% opts.numepochs =  10;
% opts.batchsize = 100;
% nn = nntrain(nn, train_x, train_y, opts);
% [er, bad] = nntest(nn, test_x, test_y);
% %disp(['Test Error: ' num2str(err)]);
% disp(['Test error is: ' num2str(er)]);
% assert(er < 0.10, 'Too big error');
