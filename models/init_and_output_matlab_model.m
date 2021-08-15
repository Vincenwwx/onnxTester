function init_and_output_matlab_model(modelName, dataRoot, savePath)
% initialize model and do transfer learning
% then output the model to specific path
% :param modelName: name of model to init
% :param dataRoot:  root path of dataset
% :param savePath:  path to save the model
% :return None

%% Basic parameters

inputMin = 0;
inputMax = 255;
numClasses = 91;    % there are 91 classes in COCO 2017


%% Initialize model
if strcmp(modelName, "inceptionv3")
    
    net = inceptionv3;
    lgraph = layerGraph(net);
    % get tid of the top layers
    lgraph = removeLayers(lgraph, ["predictions" "predictions_softmax" "ClassificationLayer_predictions"]);
    outputSizeNet = [1 1 2048];
    
    % replace the first input layer
    inputSizeNet = net.Layers(1).InputSize;
    layer = imageInputLayer(inputSizeNet, "Name", "input", "Normalization", "none");
    lgraph = replaceLayer(lgraph, 'input_1', layer);

elseif strcmp(modelName, "resnet50")
    
    net = resnet50;
    lgraph = layerGraph(net);
    lgraph = removeLayers(lgraph, ["fc1000" "fc1000_softmax" "ClassificationLayer_fc1000"]);
    outputSizeNet = [1 1 2048];
    inputSizeNet = net.Layers(1).InputSize;
    layer = imageInputLayer(inputSizeNet, "Name", "input", "Normalization", "none");
    lgraph = replaceLayer(lgraph, 'input_1', layer);

else
    net = vgg16;
    lgraph = layerGraph(net.Layers);
    lgraph = removeLayers(lgraph, ["fc8" "prob" "output"]);
    outputSizeNet = [1 1 4096];
    inputSizeNet = net.Layers(1).InputSize;
    layer = imageInputLayer(inputSizeNet, "Name", "input", "Normalization", "none");
    lgraph = replaceLayer(lgraph, 'input', layer);

end

dlnet = dlnetwork(lgraph);


%% Initialize dataset

% read coco annotation file
coco_root = fullfile(dataRoot);
filename = fullfile(coco_root, "annotations", "instances_train2017.json");
str = fileread(filename);
data = jsondecode(str);

% sort according to image id
[~,idx] = sort([data.annotations.image_id]);
data.annotations = data.annotations(idx);
data.annotations = data.annotations(1:20000);

i = 0;
j = 0;
imageIDPrev = 0;
categories = [];

while i < numel(data.annotations)
    
    i = i + 1;
    
    imageID = data.annotations(i).image_id;
    category_id = data.annotations(i).category_id;
    
    %annotationsAll = struct;
    if imageID ~= imageIDPrev
        % Create new entry
        j = j + 1;
        annotationsAll(j).ImageID = imageID;
        annotationsAll(j).Filename = fullfile(coco_root, "images", "train", pad(string(imageID),12,'left','0')+".jpg");
        annotationsAll(j).Categories = oneHotCoding(category_id, numClasses);
        categories = category_id;
    else
        % Append captions
        % annotationsAll(j).Categories = [annotationsAll(j).Categories; category_id];
        if ~ismember(category_id, categories)
            annotationsAll(j).Categories = annotationsAll(j).Categories + ...
                oneHotCoding(category_id, numClasses);
            categories = [categories, category_id];
        end
    end
    imageIDPrev = imageID;
    
end

% split the dataset: 95% training, 5% validation
% cvp = cvpartition(numel(annotationsAll), 'HoldOut', 0.05);
% idxTrain = training(cvp);
% idxTest = test(cvp);
% annotationsTrain = annotationsAll(idxTrain);
% annotationsTest = annotationsAll(idxTest);

categoriesAll = cat(1, annotationsAll.Categories);
% categoriesTrain = categoriesAll(idxTrain, :);
% categoriesTest = categoriesAll(idxTest, :);


%% Transfer learning

% prepare data for training
tblFilenames = table(cat(1, annotationsAll.Filename));
augimdsAll = augmentedImageDatastore(inputSizeNet, tblFilenames,...
    'ColorPreprocessing', 'gray2rgb');

% initialize model parameters
inputSizeClassifier = outputSizeNet(3);
parameterClassifier = struct;
parameterClassifier.fc.Weights = dlarray(initializeGlorot(numClasses, inputSizeClassifier));
parameterClassifier.fc.Bias = dlarray(zeros([numClasses 1], 'single'));

% initialize parameters for optimization
trailingAvgClassifier = [];
trailingAvgSqClassifier = [];

% specify training options
miniBatchSize = 64;
numEpochs = 2;
plots = "training-progress";
executionEnvironment = "auto"; % train on a GPU if one is available

% training
if plots == "training-progress"
    figure
    lineLossTrain = animatedline('Color', [0.85 0.325 0.098]);
    xlabel("Iteration")
    ylabel("Loss")
    ylim([0 inf])
    grid on
end

iteration = 0;
numObservationsAll = numel(annotationsAll);
%numIterationsPerEpoch = floor(numObservationsTrain / miniBatchSize);
numIterationsPerEpoch = 2;
start = tic;

% Loop over epochs.
for epoch = 1:numEpochs
    
    % Shuffle data.
    idxShuffle = randperm(numObservationsAll);
    
    % Loop over mini-batches.
    for i = 1:numIterationsPerEpoch
        iteration = iteration + 1;
        
        % Determine mini-batch indices.
        idx = (i-1)*miniBatchSize+1 : i*miniBatchSize;
        idxMiniBatch = idxShuffle(idx);
        
        % Read mini-batch of data.
        tbl = readByIndex(augimdsAll, idxMiniBatch);
        X = cat(4, tbl.input{:});
        YTrues = categoriesAll(idxMiniBatch, :);
              
        % Create batch of data.
        [dlX, dlT] = createBatch(X, YTrues, dlnet, inputMin, inputMax, executionEnvironment, modelName);
        
        % Evaluate the model gradients and loss using dlfeval and the
        % modelGradients function.
        [gradientsClassifier, loss] = dlfeval(@modelGradients, parameterClassifier, ...
            dlX, dlT);
        
        % Update encoder using adamupdate.
        [parameterClassifier, trailingAvgClassifier, trailingAvgSqClassifier] = adamupdate(parameterClassifier, ...
            gradientsClassifier, trailingAvgClassifier, trailingAvgSqClassifier, iteration);
        
        % Display the training progress.
        if plots == "training-progress"
            D = duration(0,0,toc(start),'Format','hh:mm:ss');
            addpoints(lineLossTrain,iteration,double(gather(extractdata(loss))))
            title("Epoch: " + epoch + ", Elapsed: " + string(D))
            
            drawnow
        end
    end
end

%% recombine the model

topLayers = [
    fullyConnectedLayer(numClasses,'Name','fc_out',... 
        "Weights", extractdata(parameterClassifier.fc.Weights),...
        "Bias", extractdata(parameterClassifier.fc.Bias))
    reluLayer('Name', 'relu_out')
];
lgraph = addLayers(lgraph, topLayers);
if strcmp(modelName, "inceptionv3") || strcmp(modelName, "resnet50")
    lgraph = connectLayers(lgraph,'avg_pool','fc_out');
else
    lgraph = connectLayers(lgraph,'drop7','fc_out');
end
dlnet = dlnetwork(lgraph);

%% test the model

% read coco annotation file (val)
% coco_root = fullfile(pwd, "coco_2017");
% filename = fullfile(coco_root,"annotations", "instances_val2017.json");
% str = fileread(filename);
% data = jsondecode(str);
% % sort according to image id
% [~,idx] = sort([data.annotations.image_id]);
% data.annotations = data.annotations(idx);


%% export the model to .onnx

fileName = fullfile(savePath, "matlab_"+modelName+".onnx");
exportONNXNetwork(dlnet, fileName);


end

%% Auxiliary Functions

function onehot = oneHotCoding(digit, numClasses)
    % create one-hot coding vector from a number
    % :param digit:         origin digit
    % :param numClasses:    number of classes
    % :return an one-hot encoded vector
    
    onehot = zeros(1, numClasses);
    onehot(digit) = 1;
    
end


function [dlX, dlT] = createBatch(X, YTrues, dlnet, inputMin, inputMax, executionEnvironment, modelName)
    %   create a batch dataset for training
    % :param X:                     mini-batch of data, 
    % :param documents:             tokenized captions
    % :param dlnet:                 a pretrained network
    % :param inputMin:              minimum value for image rescaling
    % :param inputMax:              maximum value for image rescaling
    % :param enc:                   a word encoding
    % :param executionEnvironment:  execution environment
    % :return a mini-batch of data corresponding to the extracted image 
    %         features and captions for training
    
    dlX = extractImageFeatures(dlnet, X, inputMin, inputMax, executionEnvironment, modelName);
    % YTrues = cat(1, YTrues{:});
    dlT = dlarray(YTrues, "BC");

    % If training on a GPU, then convert data to gpuArray.
    if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
        dlT = gpuArray(dlT);
    end

end


function dlX = extractImageFeatures(dlnet, X, inputMin, inputMax, executionEnvironment, modelName)
    % use the pretrained model to extract features
    % :param dlnet:                 a pretrained network
    % :param X:                     mini-batch of data
    % :param inputMin:              minimum value for image rescaling
    % :param inputMax:              maximum value for image rescaling
    % :param executionEnvironment:  execution environment
    % :return batch of extracted feature tensor
    
    % Resize and rescale.
    inputSize = dlnet.Layers(1).InputSize(1:2);
    X = imresize(X, inputSize);
    X = rescale(X, -1, 1, 'InputMin', inputMin, 'InputMax', inputMax);

    % Convert to dlarray.
    dlX = dlarray(X, 'SSCB');    % "spatial, spatial, channel, batch"

    % Convert to gpuArray.
    if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
        dlX = gpuArray(dlX);
    end

    % Extract features and reshape.
    dlX = predict(dlnet, dlX);
    sz = size(dlX);
    if strcmp(modelName, "inceptionv3") || strcmp(modelName, "resnet50")
        numFeatures = sz(1) * sz(2) * sz(3);
        miniBatchSize = sz(4);
        dlX = reshape(dlX, [numFeatures miniBatchSize]);
    end
    
    dlX = dlarray(dlX, "CB");

end


function [gradients, loss, dlYPred] = modelGradients(parameterClassifier, dlX, dlT)
    % calculate the gradientsd l X
    % :param topLayers:     top layers after pretrained model
    % :param dlX:           min-batch of data
    % :param dlT:           min-batch of labels
    % :return gradients     gradients
    % :return loss          loss
    % :return dlYPred       preditions

    % dlYPred = classify(dlX, parameterClassifier);
    dlYPred = fullyconnect(dlX, parameterClassifier.fc.Weights, parameterClassifier.fc.Bias);
    
    % ReLU
    dlYPred = relu(dlYPred);

    % multi-label cross entropy loss
    loss = crossentropy(dlYPred, dlT, "TargetCategories", "independent");
    
    % Calculate gradients
    gradients = dlgradient(loss, parameterClassifier);

end


function weights = initializeGlorot(numOut, numIn)
    % use Glorot initialization of weights
    % :param numOut fan-in
    % :param numIn  fan-out
    
    varWeights = sqrt( 6 / (numIn + numOut) );
    weights = varWeights * (2 * rand([numOut, numIn], 'single') - 1);

end