function energypoolkernel = GammaKernelFunction(vidFrameRate,energypoolgammashape,energypoolgammascale)
% GammaKernelFunction

    shape = energypoolgammashape;
    scale = energypoolgammascale;

    quantile = 0.99;
    div = [1, 1, 2, 6, 24, 120, 720, 5040];
    div = div(shape);
    t = 0;

    list = [];
    
    while sum(list)/vidFrameRate <= quantile
        t = t + 1/vidFrameRate;
        v = (exp(-(t/scale)) .* scale.^-shape .* t.^(-1+shape))./div;
        list = [list,v];
    end

    energypoolkernel = list;

end

