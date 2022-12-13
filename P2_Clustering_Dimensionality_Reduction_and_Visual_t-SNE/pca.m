%%
clear all; clc;
close all;

load digits.mat;

% data is already between 0 and 1
digits = digits - mean(digits); %normalizing data
% digits = digits/255;

labelCount = [];
for i = 0:9
    labelc = labels == i;
    labelCount = [labelCount sum(labelc)];
end

% sorting data and labels in increasing order
[labels,isort] = sort(labels);
digits = digits(isort,:);

trainSet = digits(1:230,:);
testSet = digits(231:460,:);
trainLabel = labels(1:230,:);
testLabel = labels(231:460,:);

trainSet = [trainSet;digits(461:745,:)];
testSet = [testSet;digits(747:1031,:)];
trainLabel = [trainLabel;labels(461:745,:)];
testLabel = [testLabel;labels(747:1031,:)];

trainSet = [trainSet;digits(1032:1296,:)];
testSet = [testSet;digits(1297:1561,:)];
trainLabel = [trainLabel;labels(1032:1296,:)];
testLabel = [testLabel;labels(1297:1561,:)];

trainSet = [trainSet;digits(1562:1811,:)];
testSet = [testSet;digits(1812:2061,:)];
trainLabel = [trainLabel;labels(1562:1811,:)];
testLabel = [testLabel;labels(1812:2061,:)];

trainSet = [trainSet;digits(2062:2311,:)];
testSet = [testSet;digits(2312:2561,:)];
trainLabel = [trainLabel;labels(2062:2311,:)];
testLabel = [testLabel;labels(2312:2561,:)];

trainSet = [trainSet;digits(2562:2789,:)];
testSet = [testSet;digits(2790:3017,:)];
trainLabel = [trainLabel;labels(2562:2789,:)];
testLabel = [testLabel;labels(2790:3017,:)];

trainSet = [trainSet;digits(3018:3248,:)];
testSet = [testSet;digits(3249:3479,:)];
trainLabel = [trainLabel;labels(3018:3248,:)];
testLabel = [testLabel;labels(3249:3479,:)];

trainSet = [trainSet;digits(3480:3735,:)];
testSet = [testSet;digits(3736:3991,:)];
trainLabel = [trainLabel;labels(3480:3735,:)];
testLabel = [testLabel;labels(3480:3735,:)];

trainSet = [trainSet;digits(3992:4235,:)];
testSet = [testSet;digits(4237:4480,:)];
trainLabel = [trainLabel;labels(3992:4235,:)];
testLabel = [testLabel;labels(3992:4235,:)];

trainSet = [trainSet;digits(4481:4740,:)];
testSet = [testSet;digits(4741:5000,:)];
trainLabel = [trainLabel;labels(4481:4740,:)];
testLabel =  [testLabel;labels(4741:5000,:)];

% PCA code
% coeffs are eigenvectors /principal component vectors. These are the eigenvectors of the covariance matrix
% scores are the projection of the data in the principal component space
% explained is percentage of variance by each eigenvalue
% coeff = eigenvectors
% score = newDataSet;
[coeff, score, latent, tsquared, explained, mu0] = pca(trainSet); %applying PCA

% Column vectors of V are eigenvector. Diagonals of D are eigenvalues
covarianceMatrix = cov(trainSet);
[V,D] = eig(covarianceMatrix);
V = fliplr(V);

dataInPrincipalComponentSpace = trainSet*coeff;

% The variances of these vectors are the eigenvalues of the covariance matrix, and are also the output "latent".
variances = var(dataInPrincipalComponentSpace)';
eigenvalues = latent;
sortedD = sort(diag(D),'descend');

figure;
stem(1:400,latent);
grid on
title("Eigenvalues in Descending Order");
ylabel("Eigenvaues");

% images for first 10 eigenvectors corresponding to highest 10 eigenvalues
for i = 1:100
    subplot(10,10,i);
    imagesc(reshape(V(:,i), 20, 20));
    colormap(gray);
    axis image;
    title("Eigenvector " + i);
    hold on;
end

% total percentage of variance contributed by top 100 eigenvalues
percentageVariance = sum(explained(1:100,:));

% displaying the sample mean for the whole training data set as an image
A = mean(trainSet);
figure;
imagesc(reshape(A,20, 20));
colormap(gray);
axis image;
title("Sample Mean for the Whole Training Data Set");

% Multiply the original data by the principal component vectors to get the projections of the original data on the
% principal component vector space. This is also the output "score".

% they are equal if data is normalized
reducDataSet = trainSet*V; %choose 99 dimensions 10*10
X_reduce = score;
% [~, pca_scores, ~, ~, var_explained] = pca(trainSet, 'NumComponents', 10);

correlation_mat = corrcoef(dataInPrincipalComponentSpace);
correlation_mat2 = corrcoef(reducDataSet);
correlation_mat3 = corrcoef(X_reduce);

% 21 different subspace dimension and projected train and testsets
trainDataSet1 = trainSet*V(:,1:1); %dimension 1
trainDataSet10 = trainSet*V(:,1:10); %dimension 10
trainDataSet20 = trainSet*V(:,1:20); %dimension 20
trainDataSet30 = trainSet*V(:,1:30); %dimension 30
trainDataSet40 = trainSet*V(:,1:40); %dimension 40
trainDataSet50 = trainSet*V(:,1:50); %dimension 50
trainDataSet60 = trainSet*V(:,1:60); %dimension 60
trainDataSet70 = trainSet*V(:,1:70); %dimension 70
trainDataSet80 = trainSet*V(:,1:80); %dimension 80
trainDataSet90 = trainSet*V(:,1:90); %dimension 90
trainDataSet100 = trainSet*V(:,1:100); %dimension 100
trainDataSet110 = trainSet*V(:,1:110); %dimension 110
trainDataSet120 = trainSet*V(:,1:120); %dimension 120
trainDataSet130 = trainSet*V(:,1:130); %dimension 130
trainDataSet140 = trainSet*V(:,1:140); %dimension 140
trainDataSet150 = trainSet*V(:,1:150); %dimension 150
trainDataSet160 = trainSet*V(:,1:160); %dimension 160
trainDataSet170 = trainSet*V(:,1:170); %dimension 170
trainDataSet180 = trainSet*V(:,1:180); %dimension 180
trainDataSet190 = trainSet*V(:,1:190); %dimension 190
trainDataSet200 = trainSet*V(:,1:200); %dimension 200

testDataSet1 = testSet*V(:,1:1); %dimension 1
testDataSet10 = testSet*V(:,1:10); %dimension 10
testDataSet20 = testSet*V(:,1:20); %dimension 20
testDataSet30 = testSet*V(:,1:30); %dimension 30
testDataSet40 = testSet*V(:,1:40); %dimension 40
testDataSet50 = testSet*V(:,1:50); %dimension 50
testDataSet60 = testSet*V(:,1:60); %dimension 60
testDataSet70 = testSet*V(:,1:70); %dimension 70
testDataSet80 = testSet*V(:,1:80); %dimension 80
testDataSet90 = testSet*V(:,1:90); %dimension 90
testDataSet100 = testSet*V(:,1:100); %dimension 100
testDataSet110 = testSet*V(:,1:110); %dimension 110
testDataSet120 = testSet*V(:,1:120); %dimension 120
testDataSet130 = testSet*V(:,1:130); %dimension 130
testDataSet140 = testSet*V(:,1:140); %dimension 140
testDataSet150 = testSet*V(:,1:150); %dimension 150
testDataSet160 = testSet*V(:,1:160); %dimension 160
testDataSet170 = testSet*V(:,1:170); %dimension 170
testDataSet180 = testSet*V(:,1:180); %dimension 180
testDataSet190 = testSet*V(:,1:190); %dimension 190
testDataSet200 = testSet*V(:,1:200); %dimension 200

%% Gaussian Classifier

% figure;
% imagesc(reshape(trainDataSet110(1,:), 10, 10));
% colormap(gray);
% axis image;

% fitting MLE parameters for Gaussian fitting

mu0 = mean(trainDataSet200(1:230,:));
mu1 = mean(trainDataSet200(231:515,:));
mu2 = mean(trainDataSet200(516:780,:));
mu3 = mean(trainDataSet200(781:1030,:));
mu4 = mean(trainDataSet200(1031:1280,:));
mu5 = mean(trainDataSet200(1281:1508,:));
mu6 = mean(trainDataSet200(1509:1739,:));
mu7 = mean(trainDataSet200(1740:1995,:));
mu8 = mean(trainDataSet200(1996:2239,:));
mu9 = mean(trainDataSet200(2240:2499,:));

% class 0
sigma0 = 0;
for i = 1:230
    sigma0 = sigma0 + (trainDataSet200(i,:)'-mu0)*(trainDataSet200(i,:)'-mu0)';
end
sigma0 = sigma0/230;
% sigma0 = abs(sigma0);

% class 1
sigma1 = 0;
for i = 231:515
    sigma1 = sigma1 + (trainDataSet200(i,:)'-mu1)*(trainDataSet200(i,:)'-mu1)';
end
sigma1 = sigma1/285;
% sigma1 = abs(sigma1);

%class 2
sigma2 = 0;
for i = 516:780
    sigma2 = sigma2 + (trainDataSet200(i,:)'-mu2)*(trainDataSet200(i,:)'-mu2)';
end
sigma2 = sigma2/265;
% sigma2 = abs(sigma2);

%class 3
sigma3 = 0;
for i = 781:1030
    sigma3 = sigma3 + (trainDataSet200(i,:)'-mu3)*(trainDataSet200(i,:)'-mu3)';
end
sigma3 = sigma3/250;
% sigma3 = abs(sigma3);

%class 4
sigma4 = 0;
for i = 1031:1280
    sigma4 = sigma4 + (trainDataSet200(i,:)'-mu4)*(trainDataSet200(i,:)'-mu4)';
end
sigma4 = sigma4/250;
% sigma4 = abs(sigma4);

%class 5
sigma5 = 0;
for i = 1281:1508
    sigma5 = sigma5 + (trainDataSet200(i,:)'-mu5)*(trainDataSet200(i,:)'-mu5)';
end
sigma5 = sigma5/228;
% sigma5 = abs(sigma5);

%class 6
sigma6 = 0;
for i = 1509:1739
    sigma6 = sigma6 + (trainDataSet200(i,:)'-mu6)*(trainDataSet200(i,:)'-mu6)';
end
sigma6 = sigma6/256;
% sigma6 = abs(sigma6);

%class 7
sigma7 = 0;
for i = 1740:1995
    sigma7 = sigma7 + (trainDataSet200(i,:)'-mu7)*(trainDataSet200(i,:)'-mu7)';
end
sigma7 = sigma7/256;
% sigma7 = abs(sigma7);

%class 8
sigma8 = 0;
for i = 1996:2239
    sigma8 = sigma8 + (trainDataSet200(i,:)'-mu8)*(trainDataSet200(i,:)'-mu8)';
end
sigma8 = sigma8/244;
% sigma8 = abs(sigma8);

%class 9
sigma9 = 0;
for i = 2240:2499
    sigma9 = sigma9 + (trainDataSet200(i,:)'-mu9)*(trainDataSet200(i,:)'-mu9)';
end
sigma9 = sigma9/260;
% sigma9 = abs(sigma9);

% testSet classification
test_class = [];

for i = 1:2499
    x = testDataSet200(i,:);
    y0 = 1/((((2*pi)^200)*det(sigma0))^1/2)*exp(-0.5*(x-mu0)*inv(sigma0)*(x-mu0'));
    y1 = 1/((((2*pi)^200)*det(sigma1))^1/2)*exp(-0.5*(x-mu1)*inv(sigma1)*(x-mu1)');
    y2 = 1/((((2*pi)^200)*det(sigma2))^1/2)*exp(-0.5*(x-mu2)*inv(sigma2)*(x-mu2)');
    y3 = 1/((((2*pi)^200)*det(sigma3))^1/2)*exp(-0.5*(x-mu3)*inv(sigma3)*(x-mu3)');
    y4 = 1/((((2*pi)^200)*det(sigma4))^1/2)*exp(-0.5*(x-mu4)*inv(sigma4)*(x-mu4)');
    y5 = 1/((((2*pi)^200)*det(sigma5))^1/2)*exp(-0.5*(x-mu5)*inv(sigma5)*(x-mu5)');
    y6 = 1/((((2*pi)^200)*det(sigma6))^1/2)*exp(-0.5*(x-mu6)*inv(sigma6)*(x-mu6)');
    y7 = 1/((((2*pi)^200)*det(sigma7))^1/2)*exp(-0.5*(x-mu7)*inv(sigma7)*(x-mu7)');
    y8 = 1/((((2*pi)^200)*det(sigma8))^1/2)*exp(-0.5*(x-mu8)*inv(sigma8)*(x-mu8)');
    y9 = 1/((((2*pi)^200)*det(sigma9))^1/2)*exp(-0.5*(x-mu9)*inv(sigma9)*(x-mu9)');
    
    y_test_vec = [y0 y1 y2 y3 y4 y5 y6 y7 y8 y9];
    [maxx1,test_ind] = max(y_test_vec);
    test_class = [test_class test_ind-1];
end

test_correct = sum(test_class == testLabel')/2499*100

% trainSet classification
train_class = [];

for j = 1:2499
    x = trainDataSet200(j,:);
    y0 = 1/((((2*pi)^200)*det(sigma0))^1/2)*exp(-0.5*(x-mu0)*inv(sigma0)*(x-mu0)');
    y1 = 1/((((2*pi)^200)*det(sigma1))^1/2)*exp(-0.5*(x-mu1)*inv(sigma1)*(x-mu1)');
    y2 = 1/((((2*pi)^200)*det(sigma2))^1/2)*exp(-0.5*(x-mu2)*inv(sigma2)*(x-mu2)');
    y3 = 1/((((2*pi)^200)*det(sigma3))^1/2)*exp(-0.5*(x-mu3)*inv(sigma3)*(x-mu3)');
    y4 = 1/((((2*pi)^200)*det(sigma4))^1/2)*exp(-0.5*(x-mu4)*inv(sigma4)*(x-mu4)');
    y5 = 1/((((2*pi)^200)*det(sigma5))^1/2)*exp(-0.5*(x-mu5)*inv(sigma5)*(x-mu5)');
    y6 = 1/((((2*pi)^200)*det(sigma6))^1/2)*exp(-0.5*(x-mu6)*inv(sigma6)*(x-mu6)');
    y7 = 1/((((2*pi)^200)*det(sigma7))^1/2)*exp(-0.5*(x-mu7)*inv(sigma7)*(x-mu7)');
    y8 = 1/((((2*pi)^200)*det(sigma8))^1/2)*exp(-0.5*(x-mu8)*inv(sigma8)*(x-mu8)');
    y9 = 1/((((2*pi)^200)*det(sigma9))^1/2)*exp(-0.5*(x-mu9)*inv(sigma9)*(x-mu9)');
    
    y_train_vec = [y0 y1 y2 y3 y4 y5 y6 y7 y8 y9];
    [maxx2,train_ind] = max(y_train_vec);
    train_class = [train_class train_ind-1];
end

train_correct = sum(train_class == trainLabel')/2499*100

test_correct_vec = [15.8156   14.7320  18.2031   19.7652   19.2136   17.2989   17.6373   17.8163   22.6146   23.1626...
23.9867   24.8165   25.1991   26.7692  31.0053   34.8694   35.5591   35.5858   35.3443  37.1057];
train_correct_vec = [15.6602   15.0104   17.0996   19.5196   19.6416   17.1391   16.5019   17.1357   22.0025   22.3490...
23.4615   23.9105   24.5881   27.0150   33.9953 35.7186   37.0585   37.2990   37.9598   39.9160];
dimension_vec = [10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200];

figure;
plot(dimension_vec,test_correct_vec);
grid on
xlabel("Feature Dimension Size");
ylabel("Accuracy");
hold on
plot(dimension_vec,train_correct_vec);
grid on
title("PCA - Success Rate of Gaussian Classifier");
xlabel("Feature Dimension Size");
ylabel("Accuracy");
legend("Test Dataset","Train Dataset");

figure;
plot(dimension_vec,100-test_correct_vec);
grid on
xlabel("Feature Dimension Size");
ylabel("Accuracy");
hold on
plot(dimension_vec,100-train_correct_vec);
grid on
title("PCA - Error Rate of Gaussian Classifier");
xlabel("Feature Dimension Size");
ylabel("Errorr");
legend("Test Dataset","Train Dataset");
