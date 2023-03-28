function VFML2 = VideoProcesser(inputVideoFilename,lumaScaler,area)
% Run video through Video Flashing Metric and mitigation

%% Initialization 

% If any input parameters do not exist, set defaults
if exist('inputVideoFilename','var') == false || isempty(inputVideoFilename), inputVideoFilename = './TestContent/TestVideo.mp4'; disp("Processing default video"); end
if exist('lumaScaler','var') == false || isempty(lumaScaler), lumaScaler = 100; disp("Assuming peak display luminance = 100 [nits]"); end
if exist('area','var') == false || isempty(area), area = 1265.625; disp("Assuming screen area size = 1265.63 [deg^2]"); end

% Load kernels and thresholds
load('./AlgorithmParameters/UFMKernels03.mat');
load('./AlgorithmParameters/UFM_PoolingNormThresholds.mat');
energyNormThreshold_table = energyNormThreshold; clear energyNormThreshold;


%% Video setup

% Input video
input_is_videofile = true;
vid = VideoReader(inputVideoFilename);
vidFrameRate = round(vid.FrameRate);
numFrames = vid.NumFrames;

% Output video
outputVideoName = 'OutputVideo.mp4';
outputVideo = VideoWriter(outputVideoName,'MPEG-4');
outputVideo.FrameRate = vidFrameRate;
outputVideo.Quality = 100;
open(outputVideo);


%% Parameters

% Inputs
gain = 1;
adaptationLevel_offset = 0.0; %[nits]
energypoolgammashape = 2.0;
energypoolexponent = 2;
energypoolgammascale = 0.15;
standardSizes = [6, 20, 45]; %[degrees]
standardNits = [0.2, 1, 10, 150, 500]; %[nits]
standardFrameRates = [24, 25, 30, 50, 60, 90, 120]; %[fps]
cA = 0.263; % Area effect on sensitivity
tauadapt = 1; % Adapting luminance time constant
taumitigation = 2.0; % Mitigation time constant

% Derived constants
muadapt = 1 - exp(-1/(tauadapt * vidFrameRate));
mumitigation = 1 - exp(-1/(taumitigation* vidFrameRate));


%% Define filters and kernels:

% Kernel frame rate
[~,closestIndexFR] = min(abs(standardFrameRates-vidFrameRate));
kernelFrameRate = standardFrameRates(closestIndexFR);

% Kernel size
equivalentSize = sqrt(area*1.6);
[~,closestIndexES] = min(abs(standardSizes-equivalentSize));
kernelSize = standardSizes(closestIndexES);

% Contrast Kernels
contrastKernels = UFMKernels(closestIndexES,:,closestIndexFR);

% Energypool machinery
energypoolkernel = GammaKernelFunction(vidFrameRate,energypoolgammashape,energypoolgammascale);
energyPoolKernelLength = length(energypoolkernel);

% Reponse adjust value:
responseAdjust = ((equivalentSize/kernelSize).^(2*cA))*(gain/vidFrameRate.^(1/energypoolexponent));

% Initialize buffers
contrastBuffers = contrastKernels;
for buffer_item = 1:length(contrastBuffers)
    contrastBuffers{buffer_item}(:) = 0; % Replace with 0s
end
energyBuffer = zeros(length(contrastBuffers),energyPoolKernelLength);
energyBuffer2 = zeros(length(contrastBuffers),energyPoolKernelLength); % Additional energy buffer

% Normalized energy threshold values:
energyNormThresholds = energyNormThreshold_table(:,:,closestIndexFR); % Load threshold values


%% Process frame:

frame_AvgLum = [];
frame_adaptationLevel = [];
frame_Contrast = [];
frame_Response = [];
frame_RiskValue = [];
frame_RiskValue2 = [];
frame_energypool = [];
frame_energypool2 = [];
frame_MitigationStrength = [];

debug = true;
debugPLOT = true;

for frame = 1:numFrames
    % RiskComputePass0 % Calculate frame APL / Apply Mitigation
    RiskComputePass0input.frameNumber = frame;
    RiskComputePass0input.MitigationStrength = frame_MitigationStrength;
    RiskComputePass0input.input_is_videofile = input_is_videofile;
    if input_is_videofile
        RiskComputePass0input.VideoReader = vid;
        RiskComputePass0input.VideoWriter = outputVideo;
        RiskComputePass0input.frame_adaptationLevel = frame_adaptationLevel;
        RiskComputePass0input.lumaScaler = lumaScaler;
    else
        RiskComputePass0input.FrameAvg = FrameAvg;
    end
    frameAvgPixelValue = RiskComputePass0( RiskComputePass0input );

    % RiskComputePass1 % Calculate Contrast 
    RiskComputePass1input.frameAvgPixelValue = frameAvgPixelValue;
    RiskComputePass1input.frame_adaptationLevel  = frame_adaptationLevel;
    RiskComputePass1input.contrastBuffers  = contrastBuffers;
    RiskComputePass1input.contrastKernels = contrastKernels;
    RiskComputePass1input.adaptationLevel_offset = adaptationLevel_offset;
    RiskComputePass1input.lumaScaler = lumaScaler;
    RiskComputePass1input.muadapt = muadapt;
    [frame_AvgLum,frame_adaptationLevel,frame_Contrast,responses,responses_normalized,contrastBuffers] = RiskComputePass1( RiskComputePass1input );

    % RiskComputePass2 % Calculate Frame Response / Risk / Mitigation Value
    RiskComputePass2input.responses = responses;
    RiskComputePass2input.responses_normalized = responses_normalized;
    RiskComputePass2input.frame_adaptationLevel = frame_adaptationLevel;
    RiskComputePass2input.energyBuffer = energyBuffer;
    RiskComputePass2input.energyBuffer2 = energyBuffer2;
    RiskComputePass2input.standardNits = standardNits;
    RiskComputePass2input.responseAdjust = responseAdjust;
    RiskComputePass2input.energypoolkernel = energypoolkernel;
    RiskComputePass2input.energypoolexponent = energypoolexponent;
    RiskComputePass2input.energyNormThresholds = energyNormThresholds;
    RiskComputePass2input.muadapt = muadapt;
    RiskComputePass2input.mumitigation = mumitigation;
    RiskComputePass2input.frame_MitigationStrength = frame_MitigationStrength;
    [frame_Energy,frame_Energy2,VFML,VFML2,frame_MitigationStrength,energyBuffer,energyBuffer2,energypool_values,energyBuffer2_values] = RiskComputePass2( RiskComputePass2input );

    if debug
        debugValue.AvgLum(frame) = frame_AvgLum;
        debugValue.adaptationLevel(frame) = frame_adaptationLevel;
        debugValue.Contrast(frame) = frame_Contrast;
        debugValue.responsesPerStandardNits(:,frame) = responses';
        debugValue.responses2PerStandardNits(:,frame) = responses_normalized';
        debugValue.energypoolPerStandardNits(:,frame) = energypool_values;
        debugValue.energypool2PerStandardNits(:,frame) = energyBuffer2_values;
        debugValue.RiskValue(frame) = VFML;
        debugValue.RiskValue2(frame) = VFML2;
        debugValue.energypool(frame) = frame_Energy;
        debugValue.energypool2(frame) = frame_Energy2;
        debugValue.MitigationStrength(frame) = frame_MitigationStrength;
        fprintf("%3d%% Processing frame %d of %d \n",round(frame/numFrames*100),frame,numFrames)
    end
end

if debug
    debugValue.area = area;
    debugValue.energypoolkernel = energypoolkernel;
    debugValue.vidFrameRate = vidFrameRate;
    debugValue.contrastKernels = contrastKernels;
end

if input_is_videofile
    close(outputVideo);
end


%% Plot
if debugPLOT
    figure
    subplot(2,1,1);
    plot(debugValue.AvgLum,'b-','Linewidth',1); grid on
    xlabel('Frame number');
    ylabel('Luminance');
    legend('APL per frame');
    subplot(2,1,2);
    plot(debugValue.RiskValue2,'r-','Linewidth',2); grid on
    ylim([0,100])
    xlabel('Frame number');
    ylabel('Risk');
    legend('Risk');
end

end
