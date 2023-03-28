//
//  main.swift
//  VideoFlashingReduction
//

import Foundation
import VideoFlashingReduction
import AVFoundation
import CoreVideo

print("Flashing Lights Detection Sample")
let detection = VideoFlashingReduction()
detection.processVideo()

while detection.isPlaying {
    CFRunLoopRunInMode(.defaultMode, 1.0, false)
}
print("Finished")

public class VideoFlashingReduction {
    var item: AVPlayerItem?
    var videoPlayer: AVPlayer?
    var detection: VideoProcessor
    var link: Timer?
    public var isPlaying: Bool = false
    
    init() {
        detection = VideoProcessor()
        detection.debugMode = true
    }
    
    func processVideo() {
        if let bundle = Bundle(identifier: "apple.VideoFlashingReduction") {
            if let url = bundle.url(forResource: "Resources/movie", withExtension: "mp4") {
                item = AVPlayerItem(asset: AVAsset(url: url))
            }
        }
        
        guard let item else {
            return
        }
        videoPlayer = AVPlayer(playerItem: item)

        guard let videoPlayer else {
            return
        }
        let videoOutput = AVPlayerItemVideoOutput(pixelBufferAttributes:
                                                    [String(kCVPixelBufferPixelFormatTypeKey): NSNumber(value: kCVPixelFormatType_32BGRA)])
        item.add(videoOutput)
        item.outputs[0].suppressesPlayerRendering = true
        
        var frame = 1
        detection.validationCallback = { (apl, risk, mitigation) in
            print("\(frame): Risk: \(risk), Mitigation: \(mitigation), APL: \(apl)")
            frame += 1
        }
        
        link = Timer.scheduledTimer(withTimeInterval: 1.0 / 60.0, repeats: true, block: { [weak self] timer in
            self?.readBuffer(timer)
        })
        isPlaying = true
        videoPlayer.play()
    }
    
    @objc func readBuffer(_ timer: Timer?) {
        if videoPlayer?.timeControlStatus != .playing {
            if videoPlayer?.timeControlStatus == .paused {
                isPlaying = false
            }
            return
        }

        guard let videoPlayer, let item else {
            isPlaying = false
            return
        }
        
        guard let avOutput = item.outputs[0] as? AVPlayerItemVideoOutput else {
            return
        }

        let itemTime = videoPlayer.currentTime()
        if avOutput.hasNewPixelBuffer(forItemTime: itemTime) {
            var outItemTimeForDisplay = CMTime()
            let pixelBuffer = avOutput.copyPixelBuffer(forItemTime: itemTime, itemTimeForDisplay: &outItemTimeForDisplay)
            guard let pixelBuffer = pixelBuffer else {
                isPlaying = false
                return
            }
            
            guard let surface = CVPixelBufferGetIOSurface(pixelBuffer)?.takeUnretainedValue() else {
                isPlaying = false
                return
            }

            guard let outSurface = IOSurfaceCreate([kIOSurfaceWidth: NSNumber(value: CVPixelBufferGetWidth(pixelBuffer)),
                                                   kIOSurfaceHeight: NSNumber(value: CVPixelBufferGetHeight(pixelBuffer)),
                                              kIOSurfacePixelFormat: NSNumber(value: kCVPixelFormatType_32BGRA),
                                              kIOSurfaceBytesPerRow: NSNumber(value: CVPixelBufferGetWidth(pixelBuffer) * 4),
                                          kIOSurfaceBytesPerElement: NSNumber(value: 4)] as CFDictionary) else {
                return
            }

            detection.processSurface(sourceSurface: surface, timestamp: CMTimeGetSeconds(outItemTimeForDisplay), destinationSurface: outSurface, options: [
                "displayMaxNits": 810,
                "sourceSurfaceEDR": 1.0,
                "displayEDRFactor": 1.0,
                "userFPS": 24.0
            ])
        }
    }
    
}
