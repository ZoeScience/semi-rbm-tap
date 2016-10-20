function test_example_SemiDBN
addpath(genpath('../'));
load mnist_uint8;

train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);

% % 10000¸öÑù±¾
% rowNumber=size(train_x,1);
% arr1=randperm(rowNumber);
% train_x=train_x(arr1(1:10000),:); 
% train_y=train_y(arr1(1:10000),:); 

% %%  ex1 train a 100 hidden unit RBM and visualize its weights
% rand('state',0)
% dbn.sizes = [100];
% opts.numepochs =   20;
% opts.batchsize = 100;
% opts.momentum  =   0.5;
% opts.alpha     =   0.005;
% opts.approx = 'tap2'
% opts.regularize=0.01
% opts.weight_decay='l1'
% opts.iterations=6
% dbn = dbnsetup(dbn, train_x, opts);
% dbn = dbntrain(dbn, train_x, opts);
% figure; visualize(dbn.rbm{1}.W');   %  Visualize the RBM weights

%%  ex2 train a 100-100 hidden unit DBN and use its weights to initialize a NN
rand('state',0)
%train dbn
% semidbn.sizes = [200];
 semidbn.sizes = [100];
opts.numepochs = 20;
 opts.batchsize = 100;
% opts.batchsize = 200;
opts.momentum  =   0.3;
 opts.alpha     =   0.05; % learning rate
% opts.alpha     =   0.08;

opts.approx = 'tap2';
opts.regularize=0.05; % regularization paramerter
opts.regularize_c= 10;

opts.weight_decay='l2';
 opts.iterations=3; % tap
 
% dbn.sizes = [100 100];
% opts.numepochs =   1;
% opts.batchsize = 100;
% opts.momentum  =   0;
% opts.alpha     =   1;
% opts.approx = 'CD'
semidbn = semidbnsetup(semidbn, train_x, opts);
semidbn = semidbntrain(semidbn, train_x, opts);
figure; visualize(semidbn.rbm{1}.W');
% saveas(gcf,'/home/leo/deepmat/myfig.jpg')




%unfold dbn to nn
nn = dbnunfoldtonn(semidbn, 10);
nn.activation_function = 'sigm';

%train nn
opts.numepochs =  20;
opts.batchsize = 100;
nn = nntrain(nn, train_x, train_y, opts);
[er, bad] = nntest(nn, test_x, test_y);
%disp(['Test Error: ' num2str(err)]);
disp(['Test error is: ' num2str(er)]);
disp(['Test Accuracy is: ' num2str((1-er)*100) '%']);
assert(er < 0.10, 'Too big error');
