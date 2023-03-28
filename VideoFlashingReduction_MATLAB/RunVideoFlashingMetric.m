%% Script to run a video through Video Flashing Metric
% User inputs
%   inputVideoFilename: path of video to be analyzed
%   lumaScaler: peak brightness of display in nits
%   area: area of display in degrees of visual angle

%%%%%%%%%%%%%%%% USER INPUT %%%%%%%%%%%%%%%%

inputVideoFilename = '';
lumaScaler = [];
area = [];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Run metric
risk = VideoProcesser(inputVideoFilename,lumaScaler,area);