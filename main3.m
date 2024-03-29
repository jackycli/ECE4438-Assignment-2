%% Assignment 2 - JD Herlehy || Jacky Li
%% Feb.29.2024
%% Creating a convolution network for the task of image classification

%%TYPE 3


%% using CIFAR10 dataset
OrganizeData;


%%options
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 64, ...
    'MaxEpochs',20,...
    'InitialLearnRate',1e-4, ...
    'Verbose',false, ...
    'Plots','training-progress');

%%layers 
layers = [
    imageInputLayer([32 32 3]) 
    convolution2dLayer(3,8, 'Padding','same') 
    batchNormalizationLayer
    reluLayer 
    convolution2dLayer(3,16, 'Padding','same') 
    batchNormalizationLayer
    reluLayer 
    convolution2dLayer(3,32, 'Padding','same') 
    batchNormalizationLayer
    reluLayer 
    convolution2dLayer(3,64, 'Padding','same') 
    batchNormalizationLayer
    reluLayer 
    convolution2dLayer(3,128, 'Padding','same') 
    batchNormalizationLayer
    reluLayer 
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

%%train
[net netInfo3] = trainNetwork(TrainingTable, layers, options);

%%test
NetworkPredict = classify(net, TestingTable(:,1));
LabelTest = TestingTable{:,2};
AccuracyTest3 = sum(NetworkPredict == LabelTest)/numel(LabelTest);

DisplayInfo(netInfo3.TrainingAccuracy(end), AccuracyTest3, 3);
