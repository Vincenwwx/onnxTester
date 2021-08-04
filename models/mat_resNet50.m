%% Classify Image using resNet
clear; close;

% Take in data
%unzip('MerchData.zip');
imds = imageDatastore('256_test', ...
                      'IncludeSubfolders',true, ...
                      'LabelSource','foldernames');
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.7, 'randomized');

% Load pretrained Network
net = resnet50;


%Extract the layer graph from the trained network and plot the layer graph.
lgraph = layerGraph(net);
% figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
%plot(lgraph)

% Check first layer input images dimensions

net.Layers(1)
inputSize = net.Layers(1).InputSize;

% Replacing last three layers for transfer learning / retraining

lgraph = removeLayers(lgraph, {'ClassificationLayer_fc1000', 'fc1000_softmax', 'fc1000'});

numClasses = numel(categories(imdsTrain.Labels));
newLayers = [
    fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];
lgraph = addLayers(lgraph,newLayers);

% Connect last transfer layer to new layers and check
lgraph = connectLayers(lgraph,'avg_pool','fc');

% figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
% %plot(lgraph)
% ylim([0,10])

% Set layers to 0 for speed and prevent over fitting

layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:110) = freezeWeights(layers(1:110));
lgraph = createLgraphUsingConnections(layers,connections);


%% Train the network
pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter,'ColorPreprocessing','gray2rgb');
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation,'ColorPreprocessing','gray2rgb');

options = trainingOptions('sgdm', ...
    'MiniBatchSize',2, ...
    'MaxEpochs',1, ... % was 6
    'InitialLearnRate',1e-4, ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'ValidationPatience',Inf, ...
    'Verbose',false, ...
    'Plots','training-progress');

[trainedNet, traininfo] = trainNetwork(augimdsTrain,lgraph,options);


%% Classify Validation Images
[YPred,probs] = classify(trainedNet,augimdsValidation);
accuracy = mean(YPred == imdsValidation.Labels)

% Display some sample images with predicted probabilities

idx = randperm(numel(imdsValidation.Files),4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label) + ", " + num2str(100*max(probs(idx(i),:)),3) + "%");
end