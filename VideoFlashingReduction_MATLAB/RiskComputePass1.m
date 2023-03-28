function [frame_AvgLum,frame_adaptationLevel,frame_Contrast,responses,responses_normalized,contrastBuffers] = RiskComputePass1(RiskComputePass1input)
% RiskComputePass1 % Calculate Contrast 

    %% Load variables
    frameAvgPixelValue = RiskComputePass1input.frameAvgPixelValue;
    frame_adaptationLevel = RiskComputePass1input.frame_adaptationLevel;
    contrastBuffers = RiskComputePass1input.contrastBuffers;
    contrastKernels = RiskComputePass1input.contrastKernels;
    adaptationLevel_offset = RiskComputePass1input.adaptationLevel_offset;
    lumaScaler = RiskComputePass1input.lumaScaler;
    muadapt = RiskComputePass1input.muadapt;

    %% Calculate contrast
    % Determine average pixel luminance by multiplying with luma scaler
    frame_AvgLum = frameAvgPixelValue * lumaScaler;
    frame_AvgLum = max(frame_AvgLum,adaptationLevel_offset); % Clamp with offset value

    % Update frame adaptation level
    if length(frame_adaptationLevel)<1 % First frame
        frame_adaptationLevel = frame_AvgLum;
    else 
        frame_adaptationLevel = muadapt * frame_AvgLum + (1-muadapt) * frame_adaptationLevel;
    end

    % Determine frame contrast
    frame_Contrast = (frame_AvgLum - frame_adaptationLevel)./frame_adaptationLevel;

    %% Update buffers
    for c = 1:length(contrastKernels)
       contrastBuffers{c} = [contrastBuffers{c}(2:end); frame_Contrast];
       
       % Filter contrast to get response
       responses(c) = sum( contrastBuffers{c}(:) .*contrastKernels{c}(end:-1:1) );
       responses2(c) = sum( contrastBuffers{c}(:) .*contrastKernels{c} );
       
       % Normalized version of contrastBuffers
       contrastBuffer_magnitude = sum( contrastBuffers{c}.^2 );
       contrastKernels_magnitude = sum( contrastKernels{c}.^2 );
       responses_normalized(c) = ( responses2(c) )./((contrastBuffer_magnitude.*contrastKernels_magnitude + 0.0000001)).^0.5;
    end

end

