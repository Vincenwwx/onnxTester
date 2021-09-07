function [fileNames, predictions, averageTime] = test_model_in_matlab(onnxPath, originFramework, dataPath, isInception)

    dsPath = fullfile(dataPath, "*.jpg");
    imds = imageDatastore(dsPath);
    runtime = 0;
    % import onnx
    params = importONNXFunction(onnxPath, 'shufflenetFcn');

    if isInception
        size = 299;
    else
        size = 224;
    end

    for i = 1 : length(imds.Files)

        fprintf('\t[MATLAB] Now test %d/%d image in MATLAB', i, length(imds.Files));
        fprintf('\n');

        [img, fileinfo] = readimage(imds, i);
        
        I = double(imresize(img, [size size], 'nearest'));
        I = rescale(I, 0, 1, 'InputMin', 0, 'InputMax', 255);
        meanIm = [0.485 0.456 0.406];
        stdIm = [0.229 0.224 0.225];
        I = (I-reshape(meanIm, [1 1 3])) ./ reshape(stdIm, [1 1 3]);
        if originFramework == "pytorch" | originFramework == "matlab"
            I = permute(I, [3 1 2]);
            I = reshape(I, [1 3 size size]);
        elseif originFramework == "tensorflow"
            I = reshape(I, [1 size size 3]);
        %elseif originFramework == "matlab"
        %    I = reshape(I, [1 3 size size]);
        end
        % predict
        tic;
        scores = shufflenetFcn(I, params, "InputDataPermutation", [1 2 3 4]);
        runtime = runtime + toc;
        % get index of top k
        %[~, re] = maxk(scores, topK);
        fileNames(i) = {fileinfo.Filename};
        %predictions(i) = {re};
        predictions(i) = {scores};
        
    end
    
    averageTime = runtime / length(imds.Files);
    
    delete("shufflenetFcn.m");  % clean-up
    
end
