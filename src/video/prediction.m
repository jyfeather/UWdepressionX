clear;clc;
addpath([ 'C:\Users\Aven\Dropbox\Aven Samareh\University of Washington\Research\Professor Shuai\AVEC 2017\YanJin\Aven\Random-Forest-Matlab-master\Random-Forest-Matlab-master']);
addpath(['C:\Users\Aven\Dropbox\Aven Samareh\University of Washington\Research\Professor Shuai\AVEC 2017\YanJin\Aven\MATLAB_scripts_functions\MATLAB_scripts_functions'])
fea_video = xlsread('Lnorm');
train = xlsread('train_split_Depression_AVEC2017.csv');
dev = xlsread('dev_split_Depression_AVEC2017.csv');
test = xlsread('test_split_Depression_AVEC2017.csv');
train_list =train(:,[1]);
dev_list =dev(:,[1]);
test_list =test(:,[1]);
video_train = fea_video(train_list ,:);
video_test = fea_video(test_list ,:);
video_dev = fea_video(dev_list ,:);
y_train = train(:,2); y_trainScore = train(:,3); 
y_dev = dev(:,2);  y_devScore = dev(:,3);
opts= struct;
opts.depth= 12;
opts.numTrees= 10;
opts.numSplits= 5;
opts.verbose= true;
opts.classifierID= 2; % weak learners to use. Can be an array for mix of weak learners too
model = forestTrain(video_train, y_trainScore, opts);
yhat = forestTest(model, video_train);

fprintf('Classifier distributions:\n');
classifierDist= zeros(1, 4);
unused= 0;
for i=1:length(model.treeModels)
    for j=1:length(model.treeModels{i}.weakModels)
        cc= model.treeModels{i}.weakModels{j}.classifierID;
        if cc>1 %otherwise no classifier was used at that node
            classifierDist(cc)= classifierDist(cc) + 1;
        else
            unused= unused+1;
        end
    end
end
fprintf('%d nodes were empty and had no classifier.\n', unused);
for i=1:4
    fprintf('Classifier with id=%d was used at %d nodes.\n', i, classifierDist(i));
end
[yhat, ysoft] = forestTest(model, video_train);
sprintf('Test accuracy: %f\n', mean(yhat==y_train));
r2 = calc_r2(y_train, yhat );
RMSE = sqrt(mean((y_trainScore - yhat).^2));  % Root Mean Squared Error
MAE = mae( y_trainScore - yhat);
[r2 Rss] = calc_r2( y_devScore , yhat );
AccuracyRate = calc_accuracyrate( y_devScore , yhat );
[ ConfusionMatrix, TP, FP, FN, TN ] = make_confusionmatrix(y_devScore , yhat );
important= xlsread('important.xlsx');
importantVariables = video_train(:,important(:,2));
corrplot(importantVariables(:,1:20));
R = corrcoef(importantVariables(:,1:20));
imagesc(R);colormap(jet);

%%

function model = forestTrain(X, Y, opts)
 
    numTrees= 100;
    verbose= false;
    
    if nargin < 3, opts= struct; end
    if isfield(opts, 'numTrees'), numTrees= opts.numTrees; end
    if isfield(opts, 'verbose'), verbose= opts.verbose; end

    treeModels= cell(1, numTrees);
    for i=1:numTrees
        
        treeModels{i} = treeTrain(X, Y, opts);
        
        % print info if verbose
        if verbose
            p10= floor(numTrees/10);
            if mod(i, p10)==0 || i==1 || i==numTrees
                fprintf('Training tree %d/%d...\n', i, numTrees);
            end
        end
    end
    
    model.treeModels = treeModels;
end

%%
function [Yhard, Ysoft] = forestTest(model, X, opts)
    
    if nargin<3, opts= struct; end
    
    numTrees= length(model.treeModels);
    u= model.treeModels{1}.classes; % Assume we have at least one tree!
    Ysoft= zeros(size(X,1), length(u));
    for i=1:numTrees
        [~, ysoft] = treeTest(model.treeModels{i}, X, opts);
        Ysoft= Ysoft + ysoft;
    end
    
    Ysoft = Ysoft/numTrees;
    [~, ix]= max(Ysoft, [], 2);
    Yhard = u(ix);
end


 