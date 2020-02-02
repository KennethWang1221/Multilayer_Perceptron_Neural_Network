%% %%== read and load UCI Classification Dataset--caesarian  ===%%
% . MLP_Perceptron Method
M = csvread('caesarian.csv.arff');

[M_shuffled] = caesarian_shuffle(M);

x_train =M_shuffled(1:5,1:50);
%x_train = (x_train)';
% x_train = x_train(:,[end,1:end-1]);
y_train =M_shuffled(6:6,1:50);


x_test =M_shuffled(1:5,51:80);
%x_test = (x_test)';
y_test =M_shuffled(6:6,51:80);

% x_test = x_test(:,[end,1:end-1]);
%% %%========== Step 1: Initialization of Multilayer Perceptron (MLP) ========
fprintf('Initializing the MLP ...\n');
n_in = 5;     % number of input neuron
n_hd1 = 20;    % number of hidden neurons
n_hd2 = 10;    % number of hidden neurons


n_ou = 1;     % number of output neuron
%w = cell(2,1); 
%w1{1} = rand(n_hd,n_in+1)./2  - 0.25; % initial weights of dim: n_hd x n_in between input layer to hidden layer
w1{1} = rand(n_hd1,n_in+1);
dw0{1}= zeros(n_hd1,n_in+1); %rand(n_hd,n_in)./2  - 0.25;%

%w1{2} = rand(n_hd,1);
w1{2} = rand(n_hd2,n_hd1+1);
%dw0{2}= zeros(n_hd,n_in+1); %rand(n_ou,n_hd)./2  - 0.25;%
dw0{2}= zeros(n_hd2,n_hd1+1); %rand(n_ou,n_hd)./2  - 0.25;%

%w1{2} = rand(n_ou,n_hd+1)./2  - 0.25; % initial weights of dim: n_ou x n_hd between hidden layer to output layer
w1{3} = rand(n_ou,n_hd2+1);

dw0{3}= zeros(n_ou,n_hd2+1); %rand(n_ou,n_hd)./2  - 0.25;%

num_Epoch = 50;      % number of epochs
mse_thres = 1E-3;    % MSE threshold
mse_train = Inf;     % MSE for training data
epoch = 1;
alpha = 0;         % momentum constant
err_point=0;
%error = 0;    % a counter to denote the number of error outputs
% eta2  = 1*ones(1,num_Epoch);         % learning-rate for output weights
% eta1  = 1*ones(1,num_Epoch);         % learning-rate for hidden weights
%eta1 = annealing(0.1,1E-5,num_Epoch);
%eta_h2 = annealing(0.1,1E-5,num_Epoch);
%eta2 = annealing(0.1,1E-5,num_Epoch);
eta1 = linspace(0.1,1E-5,num_Epoch);
eta2 = linspace(0.1,1E-5,num_Epoch);

%% %%========= Preprocess the input data : remove mean and normalize =========
mean1 = [mean(M(:,1:5))';0];
for i = 1:80,
    nor_data(:,i) = M_shuffled(:,i) - mean1;
end
max1  = [max(abs(nor_data(1:5,:)'))';1];
for i = 1:80,
    nor_data(:,i) = nor_data(:,i)./max1;
end


%% 
st = cputime;
fprintf('Training the MLP using back-propagation ...\n');
fprintf('  ------------------------------------\n');
while mse_train > mse_thres && epoch <= num_Epoch
    fprintf('   Epoch #: %d ->',epoch);
    %% shuffle the training data for every epoch
    [n_row, n_col] = size(nor_data);
    shuffle_seq = randperm(50);
    nor_data1 = nor_data(:,shuffle_seq);
   
    %% using all data for training for this epoch
    for i = 1:50,
        %% Forward computation
        x  = [nor_data1(1:5,i);1];
        %x = x';
        a = 1;
        %w1{1} = w1{1}'
        % fetching input data from database
        %d  = myint2vec(nor_data1(3,i));% fetching desired response from database
        d  = nor_data1(6,i);% fetching desired response from database
        a = 1;
        hd1 = [hyperb(w1{1}*x)];          % hidden neurons are nonlinear
        %hd1 = hd1';
        hd1  = [hd1; 1];
        a =1;
        hd2 = [hyperb(w1{2}*hd1);1]; %%%%%%second layer
        o  = hyperb(w1{3}*hd2);
        %o  = hyperb(w1{3}*hd3);         % output neuron is nonlinear
        e(:,i)  = d - o;
        
        
        
        
        
        %% Backward computation
        delta_ou = e(:,i).*d_hyperb(w1{3}*hd2);
        delta_hd2 = d_hyperb(w1{2}*hd1).*(w1{3}(:,1:n_hd2)'*delta_ou);
        delta_hd1 = d_hyperb(w1{1}*x).*(w1{2}(:,1:n_hd1)'*delta_hd2);
        dw1{1} = eta1(epoch)*delta_hd1*x';
        dw1{2} = eta1(epoch)*delta_hd2*hd1';
        dw1{3} = eta1(epoch)*delta_ou*hd2';
        %temp2 = w1{3}(:,1:n_hd2)'*delta_ou;
        %temp3 = d_hyperb(w1{2}*hd1);
        %b=2;
        %delta_ou = e(:,i).*d_hyperb(w1{2}*hd);            % delta for output layer
        %delta_hd = d_hyperb(w1{1}*x).*(w1{2}(:,1:n_hd)'*delta_ou);  % delta for hidden layer
        %dw1{1} = eta1(epoch)*delta_hd*x';
        %dw1{2} = eta2(epoch)*delta_ou*hd';
              
        %% weights update
        w2{1} = w1{1} + alpha*dw0{1} + dw1{1};  % weights input -> hidden
        w2{2} = w1{2} + alpha*dw0{2} + dw1{2};  % weights hidden-> output
        w2{3} = w1{3} + alpha*dw0{3} + dw1{3};
        
        %w2{1} = w1{1} + alpha*dw0{1} + dw1{1};  % weights input -> hidden
        %w2{2} = w1{2} + alpha*dw0{2} + dw1{2};  % weights hidden-> output
        
        %% move weights one-step
        dw0 = dw1;
        w1 = w2;
        
    end
    mse(epoch) =sum(mean(e'.^2));
    mse_train = mse(epoch);
    fprintf('MSE = %f\n',mse_train);
    epoch = epoch + 1;
end
fprintf('MSE = %f\n',mse_train);
fprintf('   Points trained : %d\n',50);
fprintf('  Epochs conducted: %d\n',epoch-1);
fprintf('        Time cost : %4.2f seconds\n',cputime - st);
fprintf('  ------------------------------------\n');
%% %%=============== Plotting Learning Curve =================================
%figure;
%plot(mse,'k');
%title('Learning curve');
%xlabel('Number of epochs');ylabel('MSE');

%% %%========================== Testing ======================================
fprintf('Testing the MLP ...\n');
nor_data2 = nor_data(:,51:80);
for i = 1:30,
    [n_row, n_col] = size(nor_data2);
    shuffle_seq = randperm(30);
    nor_data2 = nor_data2(:,shuffle_seq);
   
    
    x  = [nor_data2(1:5,i);1];
    d  = nor_data2(6,i);% fetching desired response from database
    hd1 = [hyperb(w1{1}*x)];          % hidden neurons are nonlinear
    hd1  = [hd1; 1];
    hd2 = [hyperb(w1{2}*hd1);1]; %%%%%%second layer
    o(:,i)  = hyperb(w1{3}*hd2);
    e(:,i)  = d - o(:,1);
        
    error_testing = hyperb(w1{3}*hd2);
    
    xx  = max1(1:5,:).*x(1:5,:) + mean1(1:5,:);
    d  = nor_data2(6,i);% fetching desired response from database
    %abs_o = abs(o);
    %err(:,i)  = d - abs(o);
    err(:,i)  = d - error_testing;


%     if o(:,i)>0%myvec2int(o(:,i)) == 1,
%         plot(xx(1),xx(2),'rx');
%     end
%     if o(:,i)<0%myvec2int(o(:,i)) == -1,
%         plot(xx(1),xx(2),'k+');
%     end
    mse(epoch) =sum(mean(err'.^2));
    mse_test = mse(epoch);
    fprintf('MSE = %f\n',mse_test);
    epoch = epoch + 1;

end
%abel('x');ylabel('y');
%title(['Classification using MLP with dist = ');
% Calculate testing error rate
for i = 1:30,
    %a=1
%     output4 =nor_data(6,i);
%     output6 = i;
%     output5 = o(i);
%     output3 = mysign(o(i));
%     
%     output = mysign(o(i)) - nor_data(6,i);
%     output2 = abs(mysign(o(i)) - nor_data(6,i))
%     b =2;
    if abs(mysign(o(i)) - nor_data2(6,i)) > 1E-6,
        a=1;
        err_point = err_point + 1;
    end
    
end
fprintf('  ------------------------------------\n');
fprintf('   Points tested : %d\n',30);
fprintf('    Error points : %d (%5.2f%%)\n',err_point,(err_point/30)*100);
fprintf('  ------------------------------------\n');
   
fprintf('Mission accomplished!\n');
fprintf('_________________________________________\n');



