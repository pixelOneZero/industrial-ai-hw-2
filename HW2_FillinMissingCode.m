function main()
    close all;

    %% Data Loading
    filepathFaultyUnbalanced1 = './Training/Faulty/Unbalance 1';
    filepathFaultyUnbalanced2 = './Training/Faulty/Unbalance 2';

    filepathHealthy = './Training/Healthy/';
    traindata = data_loading(filepathHealthy);
    % Append training data subfolders for faulty/unbalanced cases
    traindata = [traindata, data_loading(filepathFaultyUnbalanced1)];
    traindata = [traindata, data_loading(filepathFaultyUnbalanced2)];

    filepathTesting = './Testing/'; testdata = data_loading(filepathTesting);

    %% FFT Analysis  on Training Data
    fs = 2560; % Hz
    for k = 1:length(traindata)
        x = traindata(k).signal;
        [t, f, amp]= FFTAnalysis(x, fs);
        traindata(k).time = t;
        traindata(k).frequency = f;
        traindata(k).amplitude = amp;
    end

    %% Data Visualization on Training Data
    dataid= [1,21];  % change the sample id to visualize different data
    data_viz(traindata, dataid);

    %% Feature Extraction on Training Data
    for k = 1:length(traindata)
        dk = traindata(k);
        [feavec,feaname] = feature_extraction(dk);
        traindata(k).feature = feavec;
    end

    %% Feature Visualization - Training Data
    feature_viz(traindata,feaname);

    %% Feature Selection - Training Data
    feamat = [traindata.feature]';
    label = [traindata.label]';

% Please use the "twoclass_fisher" subfunction to compute the fisher score
% for individual features
%     **********************************************************************************************
%     **     
%     ** 
     fscore = twoclass_fisher(feamat,label);
%     **                                                                                                                              **
%     **********************************************************************************************


    figure; bar(fscore); xlabel('Feature'); ylabel('Fischer Score');

    % fscore>4;
    feaid = fscore>1;  % selected feature ID for model training
    %feamat = feamat(:,feaid);
    %% Feature Normalization - Training Data

% Please fill in the feature normalization code for the training data
% the matlab function "mapstd" is recommended
%     **********************************************************************************************
%     **                                                                                                                              ** 
    [traindataNormalized,~] = mapstd(feamat);
%     **                                                                                                                              **
%     **********************************************************************************************

    %% Model Training
    label(label==1) = 0.95; % this treatment avoids numerical instability
    label(label==0) = 0.05;
    
% Please fill in the model training code for logistic regression
% the matlab function "glmfit" is recommended
%     **********************************************************************************************
%     ** 
    [b, ~, ~] = glmfit(traindataNormalized, label, 'binomial', 'logit');
    prob = glmval(b, traindataNormalized, 'logit');
%     **                                                                                                                              **
%     **********************************************************************************************

    %% Feature Extraction - Testing Data
    for k = 1:length(testdata)
        x = testdata(k).signal;
        [t, f, amp]= FFTAnalysis(x, fs);
        testdata(k).time = t;
        testdata(k).frequency = f;
        testdata(k).amplitude = amp;

        [feavec, ~] = feature_extraction(testdata(k));
        testdata(k).feature = feavec;
    end

    testfeamat = [testdata.feature]';
    %testfeamat = testfeamat(:, feaid); % retain the useful features
    
    %% Feature Normalization  - Testing Data

% Please fill in the  feature normalization code for the testing data
% the matlab function "mapstd" is recommended
%     **********************************************************************************************
%     **                                                                                                                              **     
    [testdataNormalized, ~] = mapstd(testfeamat);
%     **                                                                                                                              **
%     **********************************************************************************************
    
    %% Model Prediction - Testing Data
% Please fill in the model testing code for logistic regression
% the matlab function "glmval" is recommended. Output of the function
% should be define as variable "cv"
%     **********************************************************************************************
%     **                                                                                                                              **
    cv = glmval(b, testdataNormalized, 'logit');
%     **                                                                                                                              **
%     **********************************************************************************************
 %   figure; plot(cv,'x-'); xlabel('Sample ID'); ylabel('Health Value')
    % number of samples of each cluster
K = length(testdataNormalized);
% offset of classes
q = 1.1;
% define 4 clusters of input data
%P = [testdataNormalized];
%P = [rand(1,K)-q rand(1,K)+q rand(1,K)+q rand(1,K)-q;
%     rand(1,K)+q rand(1,K)+q rand(1,K)-q rand(1,K)-q];
P = [transpose(testdataNormalized)-q transpose(testdataNormalized)+q transpose(testdataNormalized)+q transpose(testdataNormalized)-q;
    transpose(testdataNormalized)+q transpose(testdataNormalized)+q transpose(testdataNormalized)-q transpose(testdataNormalized)-q;]
% plot clusters
figure
plot(P(1,:),P(2,:),'k*')
hold on
grid on
%%

% SOM parameters
dimensions   = [10 10];
coverSteps   = 500;
initNeighbor = 4;
topologyFcn  = 'hextop';
distanceFcn  = 'linkdist';

% define net
net = selforgmap(dimensions,coverSteps,initNeighbor,topologyFcn,distanceFcn);
plotsomtop(net)

%%
% train
[net,Y] = train(net,P);

%%
% plot input data and SOM weight positions
plotsomtop(net)
plotsompos(net,P);
grid on

% plot SOM neighbor distances
plotsomnd(net)

% plot for each SOM neuron the number of input vectors that it classifies
figure
plotsomhits(net,P);
%% find BMU and Calculate MQE

% net.IW weight matrices of weights going to layers from network inputs
Weights = net.IW{1,1};
figure
plot(P(1,:),P(2,:),'k*')
hold on
plot(Weights(:,1),Weights(:,2),'g.')

% pick one sample
%Sample = P(:,1);
Sample = [0.5;0.5];
plot(Sample(1),Sample(2),'r*')

% find bmu
Hits = sim(net,Sample);
L = find(Hits==1);
BMU = Weights(L,:);
plot(BMU(1),BMU(2),'ro')

% MQE
MQE = norm(BMU'-Sample);
figure
plotsomhits(net,Sample);
end

function fscore=twoclass_fisher(X,y)
    uniclass = unique(y);
    
    for k = 1:2

            idxk = find( y==uniclass(k) ); nk = length(idxk);
            Xk = X(idxk,:);
            
            num(k,:) = nk.*( mean(Xk) - mean(X) ).^2;
            den(k,:) = nk.*(var(Xk) );

    end
    
    fscore = sum(num) ./ sum(den);

end

function feature_viz(data, feaname)

    feamat = [data.feature];
    label = [data.label];

    figure;
    for k = 1:length(feaname)
        ax(k) = subplot(3,3,k);

        idx1 = label == 0;
        idx2 = label == 1;
        ld(1) = plot(feamat(k,idx1), 'or'); hold on;
        ld(2) = plot(feamat(k,idx2), 'ok'); 


        xlabel('Sample ID'); ylabel(feaname{k});    
    end
    linkaxes(ax,'x');
    legend(ld,{'Faulty','Healthy'});

end

function [feavec,feaname] = feature_extraction(d)

    x = d.signal;
    f = d.frequency;
    amp = d.amplitude;
    rotatingFrequency = 20;

    % extracting statistical features: rms,  p2p, skewness, kurtosis (feel free to add other features to compare)
    feavec(1) = rms(x);
    feavec(2) = peak2peak(x);
    feavec(3) = skewness(x);
    feavec(4) = kurtosis(x);

    % Please extract the vibration features for the 1x, 2x, 3x rotating
    % frequency (the frequency range can be   1xRot +/- 5Hz)
    % **********************************************************************************************
    % **                                                                                                                              ** 

    % extracting frequency features: 1xRotating Frequency, 2xRotating
    freqRanges = rotatingFrequency * (1:3); % 1x, 2x, 3x rotating frequencies
    for i = 1:length(freqRanges)
        targetFreq = freqRanges(i);
        freqWindow = (f >= (targetFreq - 5)) & (f <= (targetFreq + 5));
        if any(freqWindow)
            %non0 = find(freqWindow, 1);
            % Calculate mean amplitude in the frequency window as a feature
            feavec(4+i) = mean(amp(freqWindow)); % min(amp(freqWindow))
        else
            feavec(4+i) = 0; % If no frequencies are in the window, set feature to 0
        end
    end


%     **
%     **                                                                                                                              **
%     **********************************************************************************************


    feavec = feavec(:);
    feaname = {'rms','peak2peak','skewness','kurtosis','1xrot','2xrot','3xrot'};
end

function [t, f, amp]= FFTAnalysis(x, fs)

    fnyq = fs/2; % Nyquist Frequency


    % Generating time axis - Define the time axis as t
    dt = 1/fs;
    N = length(x);
    t = (0:N-1).*dt;

    
    % Generating the frequency axis - Define the frequency axis as f
    df = fs/N;
    f = (0:N-1).*df;

    % Applying the FFT Function
    
    % Please refer to help documment for matlab function "fft" for more
    % information
%     **********************************************************************************************
%     **                                                                                                                              ** 
    X_fft = fft(x);
    amp = abs(X_fft/N); % N is the length of your signal x
    amp = amp(1:N/2+1);
    amp(2:end-1) = 2*amp(2:end-1);
%     **                                                                                                                              **
%     **********************************************************************************************

    
    f = f(f<=fnyq); 
    amp = amp(:); 
    f = f(:);

end

function data_viz(data, dataid)

% data - data structure
% dataid - a 2-dim vector specifies the selected data sample for
% visualization

%Generating the time and frequency scale of the signal

    d= data(dataid);

    figure;
    ax(1) = subplot(221);
    plot(d(1).time, d(1).signal); 
    xlabel('Time(Sec)'); ylabel('Health Signal'); 
    ax(2) = subplot(223);
    plot(d(2).time, d(2).signal); 
    xlabel('Time(Sec)'); ylabel('Faulty Signal'); 
    linkaxes(ax,'x');

    ax(1) = subplot(222);
    plot(d(1).frequency, d(1).amplitude); 
    xlabel('Frequency(Hz)'); ylabel('Health Spectrum'); xlim([0,200]);
    ax(2) = subplot(224);
    plot(d(2).frequency, d(2).amplitude); 
    xlabel('Frequency(Sec)'); ylabel('Faulty Spectrum');  xlim([0,200]);
    linkaxes(ax,'xy');


end

function alldata = data_loading(filepath)
    matlab_version = 0; %leave as 1 if runninng matlab 2018. change to 0 if using matlab 2021

    files = dir([filepath,'/*.txt']) ;   % you are in the folder of files
    N = length(files) ;

    % loop for each file
    for i = 1:N
        thisfile =[files(i).folder,'/', files(i).name];
        %fprintf('Converting data file %s \n', thisfile);
        T = readtable(thisfile);
        if matlab_version == 1
            T = table2array(T(5:end,1));
            T = str2double(T);
        end
        if matlab_version == 0
            T = table2array(T);
        end
        alldata(i).signal = T;
        
        fname = files(i).name;
        if strcmpi(fname(1:6), 'Normal')
            alldata(i).label = 1;
        elseif strcmpi(fname(1:11), 'Unbalance 2')
                alldata(i).label = 0;
        else
                alldata(i).label = nan;
        end

    end
    
    
end
