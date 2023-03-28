function frameAvgPixelValue = RiskComputePass0(RiskComputePass0input)
% RiskComputePass0 % Calculate frame APL / Apply Mitigation

    %% Load variables
    input_is_videofile = RiskComputePass0input.input_is_videofile;
    if input_is_videofile
        inputVideo = RiskComputePass0input.VideoReader;
        outputVideo = RiskComputePass0input.VideoWriter;
        frame_adaptationLevel = RiskComputePass0input.frame_adaptationLevel;
        lumaScaler = RiskComputePass0input.lumaScaler;
    else
        FrameAvg = RiskComputePass0input.FrameAvg;
    end
    frameNumber = RiskComputePass0input.frameNumber;
    frame_MitigationStrength = RiskComputePass0input.MitigationStrength;


    %% Linearize, color space conversion, calculate luma
    if input_is_videofile
        % Get frame average luminance
        frameRGB = double(read(inputVideo,frameNumber));
        frameRGBLin = (frameRGB./255).^2.2; %linearize
        frameRelLum = 0.299.*frameRGBLin(:,:,1)+0.587.*frameRGBLin(:,:,2)+0.114.*frameRGBLin(:,:,3);
        frameAvgPixelValue = mean(frameRelLum(:));
    else
        frameAvgPixelValue = FrameAvg(frameNumber);
    end
    

    %% Apply mitigation
    % Check if frame_MitigationStrength exists, since first frame = 0
    if input_is_videofile
        frameRGBLin_out = frameRGBLin;
        if frame_MitigationStrength % No mitigation on first frame
            imageIn = frameRGBLin;

            arlum = frame_adaptationLevel/lumaScaler;
            contrast = (1-frame_MitigationStrength);
            luminanceFactor = (1-frame_MitigationStrength);
            
            % Map output frame
            frameRGBLin_out = (min(1,max(0,(luminanceFactor*contrast*( imageIn - arlum ) + luminanceFactor*arlum))));
        end

        % Write output frame
        frameOut = uint8((frameRGBLin_out.^(1/2.2)).*255);
        writeVideo(outputVideo,frameOut);
    end


end

