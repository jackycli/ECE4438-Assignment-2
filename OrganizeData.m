%% To organize the dataset from CIFAR10

%% Training Table
% Read in the data and produce one training structure
% Store to training table
Tr1 = load('data_batch_1.mat');
Tr2 = load('data_batch_2.mat');
Tr3 = load('data_batch_3.mat');
Tr4 = load('data_batch_4.mat');
Tr5 = load('data_batch_5.mat');

trainBatchData = cat(1, Tr1.data, Tr2.data, Tr3.data, Tr4.data, Tr5.data);
trainBatchLabels = cat(1, Tr1.labels, Tr2.labels, Tr3.labels, Tr4.labels, Tr5.labels);

% Reshape the array of data to an image for each image input
% Store the image in a multidimensional array in table cell column 1
for i = 1:size(trainBatchData, 1)
    % reshape to image multidimensional array
    temp = reshape(trainBatchData(i, :), [32, 32, 3]);
    % image is reshaped to fill column first, but the data is row first
    % thus transpose
    temp = pagetranspose(temp);
    
    TrainingData(i, :) = {temp};
end

TrainingTable = table(TrainingData, categorical(trainBatchLabels));

%% Testing Table
% Read in the data and create the testing table

Ts1 = load('test_batch.mat');

for i = 1:size(Ts1.data, 1)
    % reshape to image multidimensional array
    temp = reshape(Ts1.data(i, :), [32, 32, 3]);
    % image is reshaped to fill column first, but the data is row first
    % thus transpose
    temp = pagetranspose(temp); 

    TestingData(i, :) = {temp};
end

TestingTable = table(TestingData, categorical(Ts1.labels));
