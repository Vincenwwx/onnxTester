function [fileNames, predictions, averageTime] = test_model_in_matlab(onnxPath, DataPath, isInception, topK)

    dsPath = fullfile(DataPath, "*.jpg");
    imds = imageDatastore(dsPath);
    runtime = 0;

    params = importONNXFunction(onnxPath, 'shufflenetFcn');

    if isInception
        size = 299;
    else
        size = 224;
    end

    for i = 1 : length(imds.Files)
        
        [img, fileinfo] = readimage(imds, i);
        
        I = imresize(img, [size size]);
        I = rescale(I,0,1);
        meanIm = [0.485 0.456 0.406];
        stdIm = [0.229 0.224 0.225];
        I = (I-reshape(meanIm, [1 1 3])) ./ reshape(stdIm, [1 1 3]);
        I = reshape(I, [1 size size 3]);
    
        % predict
        tic;
        scores = shufflenetFcn(I, params, "InputDataPermutation", [1 2 3 4]);
        runtime = runtime + toc;
        % get index of top k
        [~, re] = maxk(scores, topK);
        
        fileNames(i) = {fileinfo.Filename};
        predictions(i) = {re};
        
    end
    
    averageTime = runtime / length(imds.Files);
    
    delete("shufflenetFcn.m");
    
end
