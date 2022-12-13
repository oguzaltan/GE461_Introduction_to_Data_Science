clear all;
clc; close all;

%Loading Data
import_data = readtable('falldetection_dataset.csv');

%Preparing Data
dataset = table2array(import_data(:,3:end));
label_string = string(table2array(import_data(:,2)));
label_binary = label_string == "F";
label_binary = double(label_binary);

%Mean Normalization
dataset = dataset - mean(dataset);

%Splitting Data Randomly
rand_samples = randperm(566);
train_dataset = dataset(rand_samples(1:396),:);
label_train = label_binary(rand_samples(1:396),:);

valid_dataset = dataset(rand_samples(397:481),:);
label_valid = label_binary(rand_samples(397:481),:);

test_dataset = dataset(rand_samples(482:566),:);
label_test = label_binary(rand_samples(482:566),:);

%Principal Component Analysis (PCA)
[coeff, score, latent,~, explained, mu0] = pca(dataset,'NumComponents',2); %applying PCA

correl_coef = corrcoef(score);

asd = dataset*coeff;

%Plotting PCA result
% figure;
% plot(asd(:,1),asd(:,2),'.');
% grid on;
% title("Distribution of PCA Transformed Dataset in 2D");
% xlabel("First Dimension");
% ylabel("Second Dimension");

figure;
% plot(score(:,1),score(:,2),'.');
scatter(asd(:,1),asd(:,2),'.');
grid on;
title("Distribution of PCA Transformed Dataset in 2D");
xlabel("First Dimension");
ylabel("Second Dimension");

sum_variance = sum(explained(1:2));
disp("Total variance percentage contributed by top 2 PCs is " + sum_variance + "%");

cluster_dataset = asd;
cluster_dataset = score;

%K-means clustering
eva = evalclusters(score,'kmeans',"DaviesBouldin",'KList',[1:10]);
disp("Optimal number of clusters found by Davies–Bouldin index (DBI) is N = " + eva.OptimalK);

opts = statset('Display','final');
[idx,C] = kmeans(cluster_dataset,2,'Distance','cityblock',...
    'Replicates',5,'Options',opts);
figure;
plot(cluster_dataset(idx==1,1),cluster_dataset(idx==1,2),'r.','MarkerSize',12)
hold on
plot(cluster_dataset(idx==2,1),cluster_dataset(idx==2,2),'b.','MarkerSize',12)
plot(C(:,1),C(:,2),'kx',...
    'MarkerSize',15,'LineWidth',3)
grid on
legend('NF','F','Centroids',...
    'Location','NW')
title 'Cluster Assignments and Centroids'
hold off

%Checking success of k-means accuracy
figure;
silhouette(cluster_dataset,idx,"Euclidean");
title("Silhouette");

%Percentage overlap between the cluster memberships and the action labels
overlap = 0;
idx_test = idx-1;

for i = 1:size(label_binary,1)
    if idx_test(i) == label_binary(i)
        overlap = overlap+1;
    end
end

overlay_perc = overlap*100/size(label_binary,1);

if (overlay_perc  < 50)
    overlay_perc = 100-overlay_perc;
end

disp("Percentage overlap between the cluster memberships and the action labels is "...
    + overlay_perc + "%");

%Support Vector Machine (SVM)
SVMModel = fitcsvm(train_dataset,label_train);
classOrder = SVMModel.ClassNames;

sv = SVMModel.SupportVectors;
figure
gscatter(train_dataset(:,1),train_dataset(:,2),label_train)
hold on
plot(sv(:,1),sv(:,2),'ko','MarkerSize',10)
legend('F','NF','Support Vector')
title("Support Vectors and Data");
grid on
hold off

CompactSVMModel = compact(SVMModel);
% whos('SVMModel','CompactSVMModel');

%Testing SVM Model
labelz = predict(CompactSVMModel, test_dataset);
correct_test = 0;
for i = 1:85
    if(labelz(i) == label_test(i))
        correct_test = correct_test+1;
    end
end
correctness = correct_test*100/length(label_test);
disp("SVM model can predict with " + correctness + "% accuracy");

%Multilayer Perceptron (MLP)

%Network parameters
N_hidden = 100; %Number of hiddden layer neurons
std_hid = 0.01;
std_out = 0.01;
W_hid = std_hid*randn(N_hidden,(size(train_dataset,2)+1)); %Gaussian noise weights
W_out = std_out*randn(1,N_hidden+1);
learning_rate = 0.0001;

loss_vec = [];
bias = ones(1,length(train_dataset));
max_epoch = 1000;
stop_diff = 0.00001; % Threshold difference between two consecutive loss

for i = 1:max_epoch
    if i > 5
        if (loss_vec(i-2)-loss_vec(i-1)) > stop_diff
            
            %Feed forward and loss calculation
            x_hid = train_dataset;
            x_hid = [x_hid bias'];
            v_hid = W_hid*x_hid';
            y_hid = sigmoid(v_hid);
            deriv_act = sigmoid(v_hid) .* (1 - sigmoid(v_hid));
            y_hid = [y_hid; bias];
            v_out = W_out*y_hid;
            y_out = sigmoid(v_out);
            
            error = (label_train-y_out');
            loss = 1/2*(sum(error.^2));
            loss = loss/length(train_dataset);
            
            %Gradient calculation
            delta_out = y_hid*error;
            er_hid = ((error*W_out)').*(y_hid.*(1-y_hid));
            
            delta_hidden = ((er_hid(1:end-1,:))*x_hid);
            
            %Weight update
            W_out = W_out + learning_rate*delta_out';
            W_hid = W_hid + learning_rate*delta_hidden;
            
            loss_vec = [loss_vec loss]; %Storing loss for each iteration into a vector
        else
            break
        end
    else
        %Feed forward and loss calculation
        x_hid = train_dataset;
        x_hid = [x_hid bias'];
        v_hid = W_hid*x_hid';
        y_hid = sigmoid(v_hid);
        deriv_act = sigmoid(v_hid) .* (1 - sigmoid(v_hid));
        y_hid = [y_hid; bias];
        v_out = W_out*y_hid;
        y_out = sigmoid(v_out);
        
        error = (label_train-y_out');
        loss = 1/2*(sum(error.^2));
        loss = loss/length(train_dataset);
        
        %Gradient calculation
        delta_out = y_hid*error;
        er_hid = ((error*W_out)').*(y_hid.*(1-y_hid));
        
        delta_hidden = ((er_hid(1:end-1,:))*x_hid);
        
        %Weight update
        W_out = W_out + learning_rate*delta_out';
        W_hid = W_hid + learning_rate*delta_hidden;
        
        loss_vec = [loss_vec loss]; %Storing loss for each iteration into a vector
    end
end

figure;
plot(1:length(loss_vec),loss_vec);
grid on;
title("Loss Function over Iterations");
xlabel("Iterations");
ylabel("Loss");

%Validation and Hyperparameter Selection

%Network parameters
N_hidden_valid = 100; %Number of hiddden layer neurons
std_hid_valid = 0.01;
std_out_valid = 0.01;
W_hid_valid = std_hid_valid*randn(N_hidden_valid,(size(valid_dataset,2)+1)); %Gaussian noise weights
W_out_valid = std_out_valid*randn(1,N_hidden_valid+1);
learning_rate_valid = 0.001;

loss_vec_valid = [];
bias_valid = ones(1,size(valid_dataset,1));
max_epoch_valid = 1000;
stop_diff_valid = 0.00001; % Threshold difference between two consecutive loss

for i = 1:max_epoch_valid
    if i > 3
        if (loss_vec_valid(i-2)-loss_vec_valid(i-1)) > stop_diff_valid
            
            %Feed forward and loss calculation
            x_hid_valid = valid_dataset;
            x_hid_valid = [x_hid_valid bias_valid'];
            v_hid_valid = W_hid_valid*x_hid_valid';
            y_hid_valid = sigmoid(v_hid_valid);
            deriv_act_valid = sigmoid(v_hid_valid) .* (1 - sigmoid(v_hid_valid));
            y_hid_valid = [y_hid_valid; bias_valid];
            v_out_valid = W_out_valid*y_hid_valid;
            y_out_valid = sigmoid(v_out_valid);
            
            error_valid = (label_test-y_out_valid');
            loss_valid = 1/2*(sum(error_valid.^2));
            loss_valid = loss_valid/length(valid_dataset);
            
            %Gradient calculation
            delta_out_valid = y_hid_valid*error_valid;
            er_hid_valid = ((error_valid*W_out_valid)').*(y_hid_valid.*(1-y_hid_valid));
            
            delta_hidden_valid = ((er_hid_valid(1:end-1,:))*x_hid_valid);
            
            %Weight update
            W_out_valid = W_out_valid + learning_rate_valid*delta_out_valid';
            W_hid_valid = W_hid_valid + learning_rate_valid*delta_hidden_valid;
            
            loss_vec_valid = [loss_vec_valid loss_valid]; %Storing loss for each iteration into a vector
        else
            break
        end
    else
        %Feed forward and loss calculation
        x_hid_valid = valid_dataset;
        x_hid_valid = [x_hid_valid bias_valid'];
        v_hid_valid = W_hid_valid*x_hid_valid';
        y_hid_valid = sigmoid(v_hid_valid);
        deriv_act_valid = sigmoid(v_hid_valid) .* (1 - sigmoid(v_hid_valid));
        y_hid_valid = [y_hid_valid; bias_valid];
        v_out_valid = W_out_valid*y_hid_valid;
        y_out_valid = sigmoid(v_out_valid);
        
        error_valid = (label_test-y_out_valid');
        loss_valid = 1/2*(sum(error_valid.^2));
        loss_valid = loss_valid/length(valid_dataset);
        
        %Gradient calculation
        delta_out_valid = y_hid_valid*error_valid;
        er_hid_valid = ((error_valid*W_out_valid)').*(y_hid_valid.*(1-y_hid_valid));
        
        delta_hidden_valid = ((er_hid_valid(1:end-1,:))*x_hid_valid);
        
        %Weight update
        W_out_valid = W_out_valid + learning_rate_valid*delta_out_valid';
        W_hid_valid = W_hid_valid + learning_rate_valid*delta_hidden_valid;
        
        loss_vec_valid = [loss_vec_valid loss_valid]; %Storing loss for each iteration into a vector
    end
end

figure;
plot(1:length(loss_vec_valid),loss_vec_valid);
grid on;
title("Loss for MLP with Validation Data over Iterations");
xlabel("Iterations");
ylabel("Loss");

% Testing
W_hid_test = W_hid;
W_out_test = W_out;
accur = 0;

for i = 1:size(test_dataset,1)
    
    x_hid_test = test_dataset(i,:);
    x_hid_test = [x_hid_test 1];
    v_hid_test = W_hid_test*x_hid_test';
    y_hid_test = sigmoid(v_hid_test);
    y_hid_test = [y_hid_test; 1];
    v_out_test = W_out_test*y_hid_test;
    y_out_test = sigmoid(v_out_test);
    
    if y_out_test < 0.5
        if(label_test(i) == 0)
            accur = accur+1;
        end
    else
        if(label_test(i) == 1)
            accur = accur+1;
        end
    end
end

correctness = accur/size(test_dataset,1)*100;
disp("Multi Layer Perceptron model can predict with " + correctness + "% accuracy");

function sigmoidOut = sigmoid(v)
sigmoidOut = 1./(1+exp(-v));
end