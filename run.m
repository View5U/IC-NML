clear;
alg = "IC-NML";

Para.alpha   = 1e+2;
Para.beta    = 1e+1;
Para.gamma   = 1e-3;
Para.lambda  = 1e-0;
Para.lr      = 1e-6;

% mirflickr:        alpha=1e+2; beta=1e+1; gamma=1e-3; lambda=1e+0; lr=1e-6;
% music_emotion:    alpha=1e+4; beta=1e+4; gamma=1e-4; lambda=1e+1; lr=1e-7;
% music_style:      alpha=1e+3; beta=1e+2; gamma=1e-5; lambda=1e-1; lr=1e-5;
 
data_name = 'mirflickr';
load(['dataset\' data_name '.mat']);
rng();
fold = 5;

target(target==-1)=0;
N = length(target);

indices = crossvalind('Kfold',1:N,fold);
Metrics = zeros(4,fold);
for f = 1:fold
    test_idxs = (indices==f);
    train_idxs = ~test_idxs;
    train_data=data(train_idxs,:);noisy_train_target=noisy_labels(train_idxs,:);train_target=target(train_idxs,:);
    test_data=data(test_idxs,:);test_target=target(test_idxs,:);

    [train_data, settings]=mapminmax(train_data');
    test_data=mapminmax('apply',test_data',settings);
    train_data(isnan(train_data))=0;
    test_data(isnan(test_data))=0;
    train_data=train_data';
    test_data=test_data';

    fprintf("\nrunning cross validation: " + f + "\n");
    train_data(isnan(train_data))=0;
    test_data(isnan(test_data))=0;
    Para.maxIter = 50;
    Para.minLossMargin = 1e-5;
    model = IC_NML(train_data, noisy_train_target, Para); W = model.W; C = model.C;
    Ce = TuneCeffect(train_data, W, train_target, C, test_data, test_target);
    W = W*Ce;
    Outputs = (test_data*W)';
    Pre_Labels = getPredict(train_data, test_data, W, train_target, 0);

    disp('evaluating...');
    Result_MLC = EvaluationAll(Pre_Labels, Outputs, test_target');
    HammingLoss = Result_MLC(1, 1);
    AveragePrecision = Result_MLC(12, 1);
    OneError = Result_MLC(13, 1);
    RankingLoss = Result_MLC(14, 1);
    Metrics(1,f) = HammingLoss;
    Metrics(2,f) = AveragePrecision;
    Metrics(3,f) = OneError;
    Metrics(4,f) = RankingLoss;
end

Metrics = Metrics';
HammingLoss=Metrics(:,1);
AveragePrecision=Metrics(:,2);
OneError=Metrics(:,3);
RankingLoss=Metrics(:,4);
fprintf("\n" + data_name + "/" + alg + "\n");
fprintf(' HammingLoss: %f std: %f\n AveragePrecision: %f std: %f\n OneError: %f std: %f\n RankingLoss: %f std: %f\n',mean(HammingLoss),std(HammingLoss),mean(AveragePrecision),std(AveragePrecision),mean(OneError),std(OneError),mean(RankingLoss),std(RankingLoss));





%% Functions
function Pre_Labels = getPredict(train_data, test_data, W, Y, tuneThreshold)
Outputs = (test_data*W)';
if tuneThreshold == 0
    fscore     = (train_data*W);
    [tau,  ~]  = TuneThreshold(fscore', Y', 0, 1);
    Pre_Labels = Predict(Outputs,tau);
else
    Pre_Labels = double(Outputs>tuneThreshold);
end
end

function bestCe = TuneCeffect(X, W, Y, C, test_data, test_target)
probC = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 1, 1.01, 1.02, 1.05, 1.1, 1.2, 1.5, 2];
bestscore = 1;
for i=1:length(probC)
    lambda = probC(i);
    Cd = diag(diag(C));
    C0 = C - Cd;
    Ce = lambda*C0 + Cd;
    fscore = (X*W*Ce);
    Outputs = (test_data*W*Ce)';
    [tau,  ~]  = TuneThreshold(fscore', Y', 0, 1);
    Pre_Labels = Predict(Outputs, tau);
    Result = EvaluationAll(Pre_Labels, Outputs, test_target');
    score = Result(14,1);
    if score<bestscore
        bestscore = score;
        bestCe = Ce;
    end
end
end
