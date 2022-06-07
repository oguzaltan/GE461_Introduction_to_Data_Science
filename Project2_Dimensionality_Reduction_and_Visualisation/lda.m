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

% LDA Code

mu0 = mean(trainSet(1:230,:)); %class 0
mu1 = mean(trainSet(231:515,:)); %class 1
mu2 = mean(trainSet(516:780,:)); %class 2
mu3 = mean(trainSet(781:1030,:)); %class 3
mu4 = mean(trainSet(1031:1280,:)); %class 4
mu5 = mean(trainSet(1281:1508,:)); %class 5
mu6 = mean(trainSet(1509:1739,:)); %class 6
mu7 = mean(trainSet(1740:1995,:)); %class 7
mu8 = mean(trainSet(1996:2239,:)); %class 8
mu9 = mean(trainSet(2240:2499,:)); %class 9

sw0 = 0;
for i = 1:230
    sw0 = sw0+(trainSet(i,:)-mu0)'*(trainSet(i,:)-mu0);
end

sw1 = 0;
for i = 231:515
    sw1 = sw1+(trainSet(i,:)-mu1)'*(trainSet(i,:)-mu1);
end

sw2 = 0;
for i = 516:780
    sw2 = sw2+(trainSet(i,:)-mu2)'*(trainSet(i,:)-mu2);
end

sw3 = 0;
for i = 781:1030
    sw3 = sw3+(trainSet(i,:)-mu3)'*(trainSet(i,:)-mu3);
end

sw4 = 0;
for i = 1031:1280
    sw4 = sw4+(trainSet(i,:)-mu4)'*(trainSet(i,:)-mu4);
end

sw5 = 0;
for i = 1281:1508
    sw5 = sw5+(trainSet(i,:)-mu5)'*(trainSet(i,:)-mu5);
end

sw6 = 0;
for i = 1509:1739
    sw6 = sw6+(trainSet(i,:)-mu6)'*(trainSet(i,:)-mu6);
end

sw7 = 0;
for i = 1740:1995
    sw7 = sw7+(trainSet(i,:)-mu7)'*(trainSet(i,:)-mu7);
end

sw8 = 0;
for i = 1996:2239
    sw8 = sw8+(trainSet(i,:)-mu8)'*(trainSet(i,:)-mu8);
end

sw9 = 0;
for i = 2240:2499
    sw9 = sw9+(trainSet(i,:)-mu9)'*(trainSet(i,:)-mu0);
end

sw = sw0+sw1+sw2+sw3+sw4+sw5+4+sw6+sw7+sw8+sw9;

mu_all = mean(trainSet);

sb0 = 230.*(mu0-mu_all)'*(mu0-mu_all);
sb1 = 285.*(mu1-mu_all)'*(mu1-mu_all);
sb2 = 265.*(mu2-mu_all)'*(mu2-mu_all);
sb3 = 250.*(mu3-mu_all)'*(mu3-mu_all);
sb4 = 250.*(mu4-mu_all)'*(mu4-mu_all);
sb5 = 228.*(mu5-mu_all)'*(mu5-mu_all);
sb6 = 231.*(mu6-mu_all)'*(mu6-mu_all);
sb7 = 256.*(mu7-mu_all)'*(mu7-mu_all);
sb8 = 244.*(mu8-mu_all)'*(mu8-mu_all);
sb9 = 260.*(mu9-mu_all)'*(mu9-mu_all);

sb = sb0+sb1+sb2+sb3+sb4+sb5+sb6+sb7+sb8+sb9;

proj_matrix= inv(sw)*sb;
rank = rank(proj_matrix); %rank of this matrix is c-1 = 10-1 = 9
[V_lda,D_lda] = eig(proj_matrix);
V_lda = real(fliplr(V_lda));

% images of 9 bases
for i = 1:9
    subplot(3,3,i);
    imagesc(reshape(real(V_lda(:,i)), 20, 20));
    subplot(3,3,i);
    colormap(gray);
    axis image;
    title("Base " + i);
    hold on;
end

%% Gaussian Classification

%projection matrix
W1 = V_lda(:,1);
W2 = V_lda(:,1:2);
W3 = V_lda(:,1:3);
W4 = V_lda(:,1:4);
W5 = V_lda(:,1:5);
W6 = V_lda(:,1:6);
W7 = V_lda(:,1:7);
W8 = V_lda(:,1:8);
W9 = V_lda(:,1:9);

% 9 different subspace dimension and projected train and testsets
trainDataSet1 = trainSet*W1; %dimension 1
trainDataSet2 = trainSet*W2; %dimension 2
trainDataSet3 = trainSet*W3; %dimension 3
trainDataSet4 = trainSet*W4; %dimension 4
trainDataSet5 = trainSet*W5; %dimension 5
trainDataSet6 = trainSet*W6; %dimension 6
trainDataSet7 = trainSet*W7; %dimension 7
trainDataSet8 = trainSet*W8; %dimension 8
trainDataSet9 = trainSet*W9; %dimension 0

testDataSet1 = testSet*W1; %dimension 1
testDataSet2 = testSet*W2; %dimension 2
testDataSet3 = testSet*W3; %dimension 3
testDataSet4 = testSet*W4; %dimension 4
testDataSet5 = testSet*W5; %dimension 5
testDataSet6 = testSet*W6; %dimension 6
testDataSet7 = testSet*W7; %dimension 7
testDataSet8 = testSet*W8; %dimension 8
testDataSet9 = testSet*W9; %dimension 9

mu0 = mean(trainDataSet9(1:230,:));
mu1 = mean(trainDataSet9(231:515,:));
mu2 = mean(trainDataSet9(516:780,:));
mu3 = mean(trainDataSet9(781:1030,:));
mu4 = mean(trainDataSet9(1031:1280,:));
mu5 = mean(trainDataSet9(1281:1508,:));
mu6 = mean(trainDataSet9(1509:1739,:));
mu7 = mean(trainDataSet9(1740:1995,:));
mu8 = mean(trainDataSet9(1996:2239,:));
mu9 = mean(trainDataSet9(2240:2499,:));

% class 0
sigma0 = 0;
for i = 1:230
    sigma0 = sigma0 + (trainDataSet9(i,:)'-mu0)*(trainDataSet9(i,:)'-mu0)';
end
sigma0 = sigma0/230;
% sigma0 = abs(sigma0);

% class 1
sigma1 = 0;
for i = 231:515
    sigma1 = sigma1 + (trainDataSet9(i,:)'-mu1)*(trainDataSet9(i,:)'-mu1)';
end
sigma1 = sigma1/285;
% sigma1 = abs(sigma1);

%class 2
sigma2 = 0;
for i = 516:780
    sigma2 = sigma2 + (trainDataSet9(i,:)'-mu2)*(trainDataSet9(i,:)'-mu2)';
end
sigma2 = sigma2/265;
% sigma2 = abs(sigma2);

%class 3
sigma3 = 0;
for i = 781:1030
    sigma3 = sigma3 + (trainDataSet9(i,:)'-mu3)*(trainDataSet9(i,:)'-mu3)';
end
sigma3 = sigma3/250;
% sigma3 = abs(sigma3);

%class 4
sigma4 = 0;
for i = 1031:1280
    sigma4 = sigma4 + (trainDataSet9(i,:)'-mu4)*(trainDataSet9(i,:)'-mu4)';
end
sigma4 = sigma4/250;
% sigma4 = abs(sigma4);

%class 5
sigma5 = 0;
for i = 1281:1508
    sigma5 = sigma5 + (trainDataSet9(i,:)'-mu5)*(trainDataSet9(i,:)'-mu5)';
end
sigma5 = sigma5/228;
% sigma5 = abs(sigma5);

%class 6
sigma6 = 0;
for i = 1509:1739
    sigma6 = sigma6 + (trainDataSet9(i,:)'-mu6)*(trainDataSet9(i,:)'-mu6)';
end
sigma6 = sigma6/256;
% sigma6 = abs(sigma6);

%class 7
sigma7 = 0;
for i = 1740:1995
    sigma7 = sigma7 + (trainDataSet9(i,:)'-mu7)*(trainDataSet9(i,:)'-mu7)';
end
sigma7 = sigma7/256;
% sigma7 = abs(sigma7);

%class 8
sigma8 = 0;
for i = 1996:2239
    sigma8 = sigma8 + (trainDataSet9(i,:)'-mu8)*(trainDataSet9(i,:)'-mu8)';
end
sigma8 = sigma8/244;
% sigma8 = abs(sigma8);

%class 9
sigma9 = 0;
for i = 2240:2499
    sigma9 = sigma9 + (trainDataSet9(i,:)'-mu9)*(trainDataSet9(i,:)'-mu9)';
end
sigma9 = sigma9/260;
% sigma9 = abs(sigma9);

% testSet classification
test_class = [];

for i = 1:2499
    x = testDataSet9(i,:);
    y0 = 1/((((2*pi)^9)*det(sigma0))^1/2)*exp(-0.5*(x-mu0)*inv(sigma0)*(x-mu0)');
    y1 = 1/((((2*pi)^9)*det(sigma1))^1/2)*exp(-0.5*(x-mu1)*inv(sigma1)*(x-mu1)');
    y2 = 1/((((2*pi)^9)*det(sigma2))^1/2)*exp(-0.5*(x-mu2)*inv(sigma2)*(x-mu2)');
    y3 = 1/((((2*pi)^9)*det(sigma3))^1/2)*exp(-0.5*(x-mu3)*inv(sigma3)*(x-mu3)');
    y4 = 1/((((2*pi)^9)*det(sigma4))^1/2)*exp(-0.5*(x-mu4)*inv(sigma4)*(x-mu4)');
    y5 = 1/((((2*pi)^9)*det(sigma5))^1/2)*exp(-0.5*(x-mu5)*inv(sigma5)*(x-mu5)');
    y6 = 1/((((2*pi)^9)*det(sigma6))^1/2)*exp(-0.5*(x-mu6)*inv(sigma6)*(x-mu6)');
    y7 = 1/((((2*pi)^9)*det(sigma7))^1/2)*exp(-0.5*(x-mu7)*inv(sigma7)*(x-mu7)');
    y8 = 1/((((2*pi)^9)*det(sigma8))^1/2)*exp(-0.5*(x-mu8)*inv(sigma8)*(x-mu8)');
    y9 = 1/((((2*pi)^9)*det(sigma9))^1/2)*exp(-0.5*(x-mu9)*inv(sigma9)*(x-mu9)');
    
    y_test_vec = [y0 y1 y2 y3 y4 y5 y6 y7 y8 y9];
    [maxx1,test_ind] = max(y_test_vec);
    test_class = [test_class test_ind-1];
end

test_correct = sum(test_class == testLabel')/2499*100

% trainSet classification
train_class = [];

for j = 1:2499
    x = trainDataSet9(j,:);
    y0 = 1/((((2*pi)^9)*det(sigma0))^1/2)*exp(-0.5*(x-mu0)*pinv(sigma0)*(x-mu0)');
    y1 = 1/((((2*pi)^9)*det(sigma1))^1/2)*exp(-0.5*(x-mu1)*pinv(sigma1)*(x-mu1)');
    y2 = 1/((((2*pi)^9)*det(sigma2))^1/2)*exp(-0.5*(x-mu2)*pinv(sigma2)*(x-mu2)');
    y3 = 1/((((2*pi)^9)*det(sigma3))^1/2)*exp(-0.5*(x-mu3)*pinv(sigma3)*(x-mu3)');
    y4 = 1/((((2*pi)^9)*det(sigma4))^1/2)*exp(-0.5*(x-mu4)*pinv(sigma4)*(x-mu4)');
    y5 = 1/((((2*pi)^9)*det(sigma5))^1/2)*exp(-0.5*(x-mu5)*pinv(sigma5)*(x-mu5)');
    y6 = 1/((((2*pi)^9)*det(sigma6))^1/2)*exp(-0.5*(x-mu6)*pinv(sigma6)*(x-mu6)');
    y7 = 1/((((2*pi)^9)*det(sigma7))^1/2)*exp(-0.5*(x-mu7)*pinv(sigma7)*(x-mu7)');
    y8 = 1/((((2*pi)^9)*det(sigma8))^1/2)*exp(-0.5*(x-mu8)*pinv(sigma8)*(x-mu8)');
    y9 = 1/((((2*pi)^9)*det(sigma9))^1/2)*exp(-0.5*(x-mu9)*pinv(sigma9)*(x-mu9)');
    
    y_train_vec = [y0 y1 y2 y3 y4 y5 y6 y7 y8 y9];
    [maxx2,train_ind] = max(y_train_vec);
    train_class = [train_class train_ind-1];
end

train_correct = sum(train_class == trainLabel')/2499*100

% test_correct_vec = [34.7739 30.5722 21.4886 23.8095 18.3673 23.9696 24.1697 20.5282 19.7279];
% train_correct_vec = [36.8547 34.4138 20.6883 22.8091 17.0068 23.7695 22.8091 20.3281 19.0876];

test_correct_vec = [19.7279 20.5282 24.1697 23.9696 18.3673 23.8095 25.4886 30.5722 34.7739];
train_correct_vec = [19.0876 20.3281 22.8091 23.7695 17.0068 22.8091 28.6883 34.4138 36.8547];
dimension_vec = [1 2 3 4 5 6 7 8 9];

figure;
plot(dimension_vec,test_correct_vec);
grid on
xlabel("Feature Dimension Size");
ylabel("Accuracy");
hold on
plot(dimension_vec,train_correct_vec);
grid on
title("LDA - Success Rate of Gaussian Classifier");
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
title("LDA - Error Rate of Gaussian Classifier");
xlabel("Feature Dimension Size");
ylabel("Error");
legend("Test Dataset","Train Dataset");
