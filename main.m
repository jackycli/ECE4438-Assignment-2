%% Assignment 2 - JD Herlehy || Jacky Li
%% Feb.29.2024
%% Creating a convolution network for the task of image classification

%% using CIFAR10
layersRaw = [
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