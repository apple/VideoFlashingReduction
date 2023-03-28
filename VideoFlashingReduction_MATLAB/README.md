# Detection of Flashing Lights in Video Content (MATLAB)

This folder contains the MATLAB implementation of the Video Flashing Reduction
algorithm. This implementation accepts a video file (`.mp4`) as input, and outputs a 
a mitigated video and a plot of analysis results. 

A default sample video is available in `TestContent/TestVideo.mp4`. 
To run the project on your own video content, specify the input video you want to analyze.

## Run the MATLAB script

1. Open `RunVideoFlashingMetric.m` in MATLAB.
2. Set the `inputVideoFilename` variable to the path of the video you want to analyze. The default video is `TestContent/TestVideo.mp4`.
3. Set the `lumaScaler` variable to the peak brightness of your display in nits. The default value is `100`.
4. Set `area` variable to the area of your display in degrees of visual angle (for example, iPhone = `250`, Mac or larger = `1265.625`). The default value is `1265.625`.
5. Run the script.

The script writes a mitigated output video to the same folder as your input video, and a plot of results appears.

## Interpret the plot output

The output in the plot represents these values:

- APL (Average Pixel Level) [0-1] — The proportion of bright pixels in an image.
- Risk [0-100] — The risk that the current frame is flashing based on the recent sequence of frames.
