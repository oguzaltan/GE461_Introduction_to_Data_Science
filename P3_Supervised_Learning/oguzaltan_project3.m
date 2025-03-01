clear all;
clc; close all;

%Linear Regression ANN

%Preparing data
train = importdata("train1.txt");
test = importdata("test1.txt");
train_input = train(:,1)';
train_output = train(:,2)';
test_input = test(:,1)';
test_output = test(:,2)';

%Normalization
train_input = train_input - mean(train_input);
train_output = train_output - mean(train_output);

%Displaying scatter matrix of original data
figure;
scatter(train_input,train_output);
grid on
title("Scatter Matrix of Training Data");
xlabel("Training Input Data");
ylabel("Training Output Data");

figure;
scatter(test_input,test_output);
grid on
title("Scatter Matrix of Test Data");
xlabel("Test Input Data");
ylabel("Test Output Data");

%Initialize weights and parameters
std = 0.01;
W = std*randn(1,2); %Standart normal distribution

learning_rate = 0.0001; %Learning rate
loss_vec = [];

rand_index = randperm(60); %Randomly shuffled indexes
train_input = train_input(rand_index);
train_output = train_output(rand_index);

bias = ones(1,60);
train_input = [train_input; bias]; %Adding bias to the weights matrix

%Learning
max_epoch = 1000; %Number of maximum iterations
stop_diff = 0.0001; %Threshold difference between two consecutive loss

for i = 1:max_epoch
    if i > 2
        if (loss_vec(i-2)-loss_vec(i-1)) > stop_diff
            y = W*train_input; %Weighted sum
            error = (train_output-y); %Difference between the desired and realized output of the network
            loss = 1/2*(sum(error.^2));
            loss = loss/60;
            W = W + learning_rate*error*train_input';
            loss_vec = [loss_vec loss]; %Storing loss for each iteration into a vector
        else
            break
        end
    else
        y = W*train_input; %Weighted sum
        error = (train_output-y); %Difference between the desired and realized output of the network
        loss = 1/2*(sum(error.^2));
        loss = loss/60;
        W = W + learning_rate*error*train_input';
        loss_vec = [loss_vec loss]; %Storing loss for each iteration into a vector
    end
end

figure;
plot(1:length(loss_vec),loss_vec);
grid on;
title("Loss Function Curve for Linear Regressor over Iterations");
xlabel("Iterations");
ylabel("Loss");

figure;
scatter (train_input(1,:),train_output);
grid on;
hold on;

input_po = -7.6490:0.001:10.5077;
curve = W(1)*input_po+W(2);

save ("curvee.mat","curve","-ascii");

plot (input_po,curve);
title("Linear Regressor Line Fit for Training Data");
xlabel("Training Set Input");
ylabel("Training Set Output");

%Testing
W_test = W;

figure;
scatter(test_input(1,:),test_output);
grid on;
hold on;

input_po_test = -9.6429:0.001:8.9747;
curve_test = W_test(1)*input_po_test + W_test(2);
plot (input_po_test,curve_test);
title("Linear Regressor Line Fit for Test Data");
xlabel("Test Set Input");
ylabel("Test Set Output");

%Calculating test error
curve_test = W_test(1)*test_input + W_test(2);
error = curve_test - test_input;
loss_test = 1/2*(sum(error.^2));
loss_test = loss_test/41;

%% ANN with Hidden Layer
clear all;
clc; close all;

%Preparing data
train = importdata("train1.txt");
test = importdata("test1.txt");
train_input = train(:,1)';
train_output = train(:,2)';
test_input = test(:,1)';
test_output = test(:,2)';

%Normalization
train_input = train_input - mean(train_input);
train_output = train_output - mean(train_output);

%Stochastic learning
rand_index = randperm(60); %Randomly shuffled indexes
train_input = train_input(rand_index);
train_output = train_output(rand_index);

%Network parameters
N_hidden = 100; %Number of hiddden layer neurons
std_hid = 0.01;
std_out = 0.01;
W_hid = std_hid*randn(N_hidden,2); %Gaussian noise weights
W_out = std_out*randn(1,N_hidden+1);
learning_rate = 0.00001;
% learning_rate = 0.00000001;
loss_vec = [];
bias = ones(1,60);
max_epoch = 10000;
stop_diff = 0; % Threshold difference between two consecutive loss

for i = 1:max_epoch
    if i > 10
        if (loss_vec(i-2)-loss_vec(i-1)) > stop_diff
            
            %Feed forward and loss calculation
            x_hid = train_input;
            x_hid = [x_hid; bias];
            v_hid = W_hid*x_hid;
            y_hid = sigmoid(v_hid);
            deriv_act = sigmoid(v_hid) .* (1 - sigmoid(v_hid));
            y_hid = [y_hid; bias];
            v_out = W_out*y_hid;
            y_out = v_out;
            error = (train_output-y_out);
            loss = 1/2*(sum(error.^2));
            loss = loss/60;
            
            %Gradient calculation
            delta_out = error*y_hid';
            er_hid = (W_out'*error).*(y_hid.*(1-y_hid));
            delta_hidden = ((er_hid(1:end-1,:))*x_hid');
            
            %Weight update
            W_out = W_out + learning_rate*delta_out;
            W_hid = W_hid + learning_rate*delta_hidden;
            
            loss_vec = [loss_vec loss]; %Storing loss for each iteration into a vector
        else
            break
        end
    else
        %Feed forward and loss calculation
        x_hid = train_input;
        x_hid = [x_hid; bias];
        v_hid = W_hid*x_hid;
        y_hid = sigmoid(v_hid);
        deriv_act = sigmoid(v_hid) .* (1 - sigmoid(v_hid));
        y_hid = [y_hid; bias];
        v_out = W_out*y_hid;
        y_out = v_out;
        error = (train_output-y_out);
        loss = 1/2*(sum(error.^2));
        loss = loss/60;
        
        %Gradient calculation
        delta_out = error*y_hid';
        er_hid = (W_out'*error).*(y_hid.*(1-y_hid));
        delta_hidden = ((er_hid(1:end-1,:))*x_hid');
        
        %Weight update
        W_out = W_out + learning_rate*delta_out;
        W_hid = W_hid + learning_rate*delta_hidden;
        
        loss_vec = [loss_vec loss]; %Storing loss for each iteration into a vector
    end
end

figure;
plot(1:length(loss_vec),loss_vec);
grid on;
title("Loss Function Curve for Linear Regressor over Iterations");
xlabel("Iterations");
ylabel("Loss");

%Regression on test dataset
input_po_test = -9.6429:0.1:8.9747; %Choosing points in the range of input data

x_hid_test = zeros(1,length(input_po_test));
y_out_test = zeros(1,length(input_po_test));

%Feed Forwarding Test Data in the Network
for i = 1:length(input_po_test)
    x = input_po_test(i);
    x = [x;1];
    v_hid = W_hid*x;
    y_hid = sigmoid(v_hid);
    y_hid = [y_hid; 1];
    v_out = W_out*y_hid;
    y_out_test(i) = v_out;
end

%Visualising Scatter Matrices
figure;
scatter(test_input(1,:),test_output);
grid on;
hold on;

scatter (input_po_test,y_out_test);
title("Single Hidden Layer ANN Regressor for Test Data");
xlabel("Test Set Input");
ylabel("Test Set Output");

%Regression on training dataset
input_po_train = -7.6490:0.1:10.5077; %Choosing points in the range of input data

x_hid_test = zeros(1,length(input_po_train));
y_out_test = zeros(1,length(input_po_train));

for i = 1:length(input_po_train)
    x = input_po_train(i);
    x = [x;1];
    v_hid = W_hid*x;
    y_hid = sigmoid(v_hid);
    y_hid = [y_hid; 1];
    v_out = W_out*y_hid;
    y_out_train(i) = v_out;
end

%Visualising Scatter Matrices
figure;
scatter(train_input(1,:),train_output);
grid on;
hold on;

scatter (input_po_train,y_out_train);
title("Single Hidden Layer ANN Regressor for Training Data");
xlabel("Training Set Input");
ylabel("Training Set Output");

%Hidden Layer Neurons - ANN Complexity
curvee = load("curvee.mat","-ascii");
y_out2 = load("hid2out.mat","-ascii");
y_out4 = load("hid4out.mat","-ascii");
y_out8 = load("hid8out.mat","-ascii");
y_out16 = load("hid16out.mat","-ascii");
y_out32 = load("hid32out.mat","-ascii");

%Visualising Scatter Matrices
figure;
scatter(train_input(1,:),train_output);
grid on;
hold on;

scatter (input_po_train,y_out2);
scatter (input_po_train,y_out4);
scatter (input_po_train,y_out8);
scatter (input_po_train,y_out16);
scatter (input_po_train,y_out32);

input_po = -7.6490:0.001:10.5077;

plot (input_po,curvee);

title("Single Hidden Layer ANN Regressor for Training Data");
xlabel("Training Set Input");
ylabel("Training Set Output");
legend("Nhidden = 32","Nhidden = 2","Nhidden = 4","Nhidden = 8","Nhidden = 16");

function sigmoidOut = sigmoid(v)
sigmoidOut = 1./(1+exp(-v));
end