# Detection of Flashing Lights in Video Content (Xcode)

This folder contains the Xcode implementation of the Video Flashing Reduction
algorithm. This implementation accepts a video file (`.mp4`) as input, and outputs a 
frame-by-frame analysis of the risk of flashing lights in a particular frame of that video.

A default sample video is available in `Resources/movie.mp4`. To run the project on your 
own video content, replace the sample video with the video you want to analyze.

## Run the project in Xcode

1. Open `VideoFlashingReduction.xcodeproj` in Xcode.
2. In the Scheme menu, choose `detectflashing`.
3. Choose My Mac as the run destination.
4. Press Command-R to run the project.

The frame-by-frame analysis outputs to the logging area in Xcode.

## Run the project in the command line
 
In the Terminal, run `detectflashing` and provide the `.mp4` file you want to analyze as input:

```
detectflashing movie.mp4
```

The frame-by-frame analysis outputs to the command line.

## Interpret the analysis output

The output from the analysis is:

- APL (Average Pixel Level) [0-1] — The proportion of bright pixels in an image.
- Risk [0-1] — The risk that the current frame is flashing based on the recent sequence of frames.
- Mitigation [0-1] — How much mitigation will be applied to the out surface to reduce risk.
