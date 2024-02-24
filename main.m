%% Assignment 2 - JD Herlehy || Jacky Li
%% Feb.29.2024
%% Creating a convolution network for the task of image classification

%% using CIFAR10

options = trainingOptions('sgdm', ...
    'MaxEpochs',300,...
    'InitialLearnRate',2e-3, ...
    'Verbose',false, ...
    'Plots','training-progress');

layers = [
    imageInputLayer([32 32 3])...
    convolution2dLayer(3,8)
    reluLayer
    convolution2dLayer(3,16)
    reluLayer
    convolution2dLayer(3,32)
    reluLayer
    convolution2dLayer(3,64)
    reluLayer
    convolution2dLayer(3,128)
    reluLayer
    fullyConnectedLayer(10)
    classificationLayer];


net = trainNetwork(,layers,options);