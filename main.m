%% Assignment 2 - JD Herlehy || Jacky Li
%% Feb.29.2024
%% Creating a convolution network for the task of image classification

%% using CIFAR10 dataset
numImages = size(data,1);

%matching numerical values to categories
labels_categorical = label_names(labels + 1);

% Define the dimensions of the images
imageHeight = 32;
imageWidth = 32;
numChannels = 3;

% Reshape the images to a 32x32x3x10000 array
reshaped_images = reshape(data', [imageWidth, imageHeight, numChannels, numImages]);

% Rearrange the dimensions to have the images in the correct order
reshaped_images = permute(reshaped_images, [2, 1, 3, 4]);

% Convert the array to a 1-D cell array
cellArrayImages = cell(numImages, 1);

for i = 1:numImages
    cellArrayImages{i} = reshaped_images(:, :, :, i);
end

trainData = [cellArrayImages, labels_categorical];

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
    fullyConnectedLayer(10)
    classificationLayer];


net = trainNetwork(trainData,layers,options);