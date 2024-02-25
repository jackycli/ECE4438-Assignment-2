%% Assignment 2 - JD Herlehy || Jacky Li
%% Feb.29.2024
%% Creating a convolution network for the task of image classification

%% using CIFAR10 dataset
numImages = size(data,1);

%%matching numerical values to categories
labels_categorical = label_names(labels + 1);
counter = (1:numImages)';
for i = 1:numImages
    % Assign data and label to each cell
    TableTrain{i, 1} = data(counter(i), :);
    TableTrain{i, 2} = labels_categorical(counter(i));
end

options = trainingOptions('sgdm', ...
    'MaxEpochs',20,...
    'InitialLearnRate',1e-4, ...
    'Verbose',false, ...
    'Plots','training-progress');

layers = [
    imageInputLayer([32 32 3])
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


net = trainNetwork(TableTrain,layers,options);