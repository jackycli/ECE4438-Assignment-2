%% Assignment 2 - JD Herlehy || Jacky Li
%% Feb.29.2024
%% Creating a convolution network for the task of image classification

%% using CIFAR10 dataset
OrganizeData;

%Type 2

%%options
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 64, ...
    'MaxEpochs',20,...
    'InitialLearnRate',1e-4, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor', 0.2000, ...
    'LearnRateDropPeriod', 5, ...
    'Verbose',false, ...
    'Plots','training-progress');

%%layers 
layers = [
    imageInputLayer([32 32 3]) 
    convolution2dLayer(3,8, 'Padding','same') 
    batchNormalizationLayer
    reluLayer 
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,16, 'Padding','same') 
    batchNormalizationLayer
    reluLayer 
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,32, 'Padding','same') 
    batchNormalizationLayer
    reluLayer 
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,64, 'Padding','same') 
    batchNormalizationLayer
    reluLayer 
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,128, 'Padding','same') 
    batchNormalizationLayer
    reluLayer 
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

%%train
net = trainNetwork(TrainingTable, layers, options);

%%test
NetworkPredict = classify(net, TestingTable(:,1));
LabelTest = TestingTable{:,2};
AccuracyTest = sum(NetworkPredict == LabelTest)/numel(LabelTest);
disp("Testing Accuracy:")
disp(AccuracyTest)
