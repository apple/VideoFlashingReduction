function [frame_Energy,frame_Energy2,VFML,VFML2,frame_MitigationStrength,energyBuffer,energyBuffer2,energypool_values,energyBuffer2_values] = RiskComputePass2(RiskComputePass2input)
% RiskComputePass2 % Calculate Frame Response / Risk / Mitigation Value
   
    %% Load variables
    responses = RiskComputePass2input.responses;
    responses_normalized = RiskComputePass2input.responses_normalized;
    frame_adaptationLevel = RiskComputePass2input.frame_adaptationLevel;
    energyBuffer = RiskComputePass2input.energyBuffer;
    energyBuffer2 = RiskComputePass2input.energyBuffer2;
    standardNits = RiskComputePass2input.standardNits;
    responseAdjust = RiskComputePass2input.responseAdjust;
    energypoolkernel = RiskComputePass2input.energypoolkernel;
    energypoolexponent = RiskComputePass2input.energypoolexponent;
    energyNormThresholds= RiskComputePass2input.energyNormThresholds;
    previous_MitigationStrength = RiskComputePass2input.frame_MitigationStrength;
    muMitigation = RiskComputePass2input.mumitigation;
    

    %% Update energy buffers
    num_kernels = length(responses);
    energypool_values = zeros(num_kernels,1);
    energyBuffer2_values = zeros(num_kernels,1);
    for nitsLevel = 1:num_kernels
        % Update (last value of) energy buffers
        energyBuffer(nitsLevel,:) = [energyBuffer(nitsLevel,2:end),responses(nitsLevel)^energypoolexponent];
        energyBuffer2(nitsLevel,:) = [energyBuffer2(nitsLevel,2:end),responses_normalized(nitsLevel).^energypoolexponent];

        energypool_values(nitsLevel) = sum(energyBuffer(nitsLevel,:).*fliplr(energypoolkernel)).^(1/energypoolexponent);
        energyBuffer2_values(nitsLevel) = sum(energyBuffer2(nitsLevel,:).*fliplr(energypoolkernel)).^(1/energypoolexponent);
    end


    %% Interpolate energies:
    logNits = log10(frame_adaptationLevel);
    logStandardNits = log10(standardNits);
    if logNits < logStandardNits(1)
        frame_Energy = energypool_values(1);
        frame_Energy2 = energyBuffer2_values(1);
    elseif logNits < logStandardNits(2)
        frame_Energy = energypool_values(1) + (logNits - logStandardNits(1)) * (energypool_values(2) - energypool_values(1))/(logStandardNits(2) - logStandardNits(1));
        frame_Energy2 = energyBuffer2_values(1) + (logNits - logStandardNits(1)) * (energyBuffer2_values(2) - energyBuffer2_values(1))/(logStandardNits(2) - logStandardNits(1));
    elseif logNits < logStandardNits(3)
        frame_Energy = energypool_values(2) + (logNits - logStandardNits(2)) * (energypool_values(3) - energypool_values(2))/(logStandardNits(3) - logStandardNits(2));
        frame_Energy2 = energyBuffer2_values(2) + (logNits - logStandardNits(2)) * (energyBuffer2_values(3) - energyBuffer2_values(2))/(logStandardNits(3) - logStandardNits(2));
    elseif logNits < logStandardNits(4)
        frame_Energy = energypool_values(3) + (logNits - logStandardNits(3)) * (energypool_values(4) - energypool_values(3))/(logStandardNits(4) - logStandardNits(3));
        frame_Energy2 = energyBuffer2_values(3) + (logNits - logStandardNits(3)) * (energyBuffer2_values(4) - energyBuffer2_values(3))/(logStandardNits(4) - logStandardNits(3));
    elseif logNits < logStandardNits(5)
        frame_Energy = energypool_values(4) + (logNits - logStandardNits(4)) * (energypool_values(5) - energypool_values(4))/(logStandardNits(5) - logStandardNits(4));
        frame_Energy2 = energyBuffer2_values(4) + (logNits - logStandardNits(4)) * (energyBuffer2_values(5) - energyBuffer2_values(4))/(logStandardNits(5) - logStandardNits(4));
    else
        frame_Energy = energypool_values(5);
        frame_Energy2 = energyBuffer2_values(5);
    end
    frame_Energy = frame_Energy .* responseAdjust;
    frame_Energy2 = energyBuffer2_values(5);

    
    %% Map energy pool to 0-100 risk range
    location = 33;
    scale = 200;
    shape = 3;
    VFML = 0;
    if frame_Energy>location
        VFML = 100.*(1 - exp(-( (-location + frame_Energy)./scale ).^shape ));
    end
    frame_RiskValue = VFML;
    
    %% Step masking
    % Determine masking threshold of risk value
    [~,closestIndexLum] = min(abs( logStandardNits - log10(frame_adaptationLevel+0.00001) ));
    energypool_threshold = energyNormThresholds(1,closestIndexLum);
    VFML2 = 0;
    if frame_Energy2 > 1.8 % energypool_threshold
        VFML2 = VFML;
    end
    frame_RiskValue = VFML2;


    %% Calculate mitigation strength
    gain = 0.5; % Maps Risk = 100 -> Mitigation = 1.0
    mitigationThreshold = 0;
    if (frame_RiskValue > mitigationThreshold)
        frame_MitigationStrength = max( 0, min( 1 , log10(frame_RiskValue)*gain ));
    else 
        frame_MitigationStrength = 0;
    end

    if (frame_MitigationStrength < previous_MitigationStrength) % mitigationStrength of frame i is lower than that of previous frame i-1
        frame_MitigationStrength = frame_MitigationStrength*muMitigation + previous_MitigationStrength*(1 - muMitigation);
        if frame_MitigationStrength < 0.01
            frame_MitigationStrength = 0;
        end
    end
end

