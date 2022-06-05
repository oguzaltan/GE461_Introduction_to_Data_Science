clear all; clc;
close all;

load digits.mat;

labelCount = [];
for i = 0:9
    labelc = labels == i;
    labelCount = [labelCount sum(labelc)];
end

% sorting data and labels in increasing order
[labels,isort] = sort(labels);
digits = digits(isort,:);

% rng default % for reproducibility
rng default % for reproducibility
Y = tsne(digits,'Algorithm','barneshut','NumPCAComponents',50);
gscatter(Y(:,1),Y(:,2),labels);
title("2D Scatter Matrix with Class Informations - tSNE");
xlabel("First Dimension");
ylabel("Second Dimension");