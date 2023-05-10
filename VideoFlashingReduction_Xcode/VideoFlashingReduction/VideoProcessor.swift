//
//  VideoProcessor.swift
//  VideoFlashingReduction
//

import Foundation
import Metal
import os
import CoreVideo
@_implementationOnly import VideoFlashingReduction_Bridge_Internal

public final class VideoProcessor: NSObject {
    
    struct KernelData {
        public var length: [UInt] = Array(repeating: 0, count: Int(numAdaptationConditions))
        public var array: [Float] = Array()
    }
    
    struct ArrayData {
        public var length: UInt
        public var array: [Float] = Array()
    }
    
    private var needsInitialization: Bool = true
    private var device: MTLDevice?
    private var commandQueue: MTLCommandQueue?
    private var library: MTLLibrary?
    private var previousSurfaceTime: CFAbsoluteTime = 0
    
    private var cptPSO_RiskComputePass0: MTLComputePipelineState?
    private var cptPSO_RiskComputePass1: MTLComputePipelineState?
    private var cptPSO_RiskComputePass2: MTLComputePipelineState?
    private var cptPSO_RiskComputePass3: MTLComputePipelineState?
    
    private var bufferFrameLumaSum: MTLBuffer?
    private var bufferData: MTLBuffer?
    private var bufferCurState: MTLBuffer?
    private var bufferGammaKernel: MTLBuffer?
    private var bufferEnergy: MTLBuffer?
    private var bufferEnergy2: MTLBuffer?
    private var bufferContrastKernel: MTLBuffer?
    private var bufferContrast: MTLBuffer?
    private var bufferResponses: MTLBuffer?
    private var bufferResponsesNorm: MTLBuffer?
    private var bufferResults: MTLBuffer?
    private var bufferDataDebug: MTLBuffer?

    private var sourceTexture: MTLTexture?
    private var processedTexture: MTLTexture? = nil
    private var bufferConstants: ConstBuf
    
    private var fps: Float = 60.0
    private var nits: Float = 0
    private var area: Float = 1265.63
    private var avl: Float = 0.15
    private var gain: Float = 1.0
    private var energyPoolGammaShape: Float = 2.0
    private var energyPoolGammaScale: Float = 0.15
    private var probabilityPoolGammaShape: Float = 4.0
    private var probabilityPoolExponent: Float = 4.0
    private var cA: Float = 0.263
    private var tauAdapt: Float = 1.0
    private var tauMitigation: Float = 2.0

    private var idxFrameRate: Int = 0
    private var idxEquivalentSize: Int = 0
    private var idxEquivalentKernelIndex: Int = 0
    
    private var frameDeltas: [Double]
    private var framePoolIndex: Int = 0

    public var validationCallback: ((Float, Float, Float) -> Void)?
    public var inTestingMode: Bool = false

    public var debugMode: Bool = false {
        didSet {
            bufferConstants.bDebug = self.debugMode
        }
    }

    @objc public override init() {
        frameDeltas = Array(repeating: 1.0 / 30.0, count: Int(FPS_Pool_Cnt))
        bufferConstants = ConstBuf()
        super.init()
    }
    
    private func initialize() {
        UMFKernels().initUMFKernels()

        device = MTLCreateSystemDefaultDevice()
        guard let device = device else {
            return
        }

        commandQueue = device.makeCommandQueue()

        let scope = MTLCaptureManager.shared().makeCaptureScope(device: device)
        MTLCaptureManager.shared().defaultCaptureScope = scope

        do {
            try library = device.makeDefaultLibrary(bundle: Bundle(for: VideoProcessor.self))
        } catch {
            os_log("Error making library: %@", log: .default, type: .error, error.localizedDescription)
        }

        initializeBuffers()

        bufferDataDebug = device.makeBuffer(length: Int(DEVAR_Cnt) * MemoryLayout<Float>.stride, options: .storageModeShared)

        resetState()

        if let cs = library?.makeFunction(name: "cs_compute_risk_pass0") {
            do {
                try cptPSO_RiskComputePass0 = device.makeComputePipelineState(function: cs)
            } catch {
                os_log("Error creating cs_compute_risk_pass0: %@", log: .default, type: .error, error.localizedDescription)
            }
        }

        if let cs = library?.makeFunction(name: "cs_compute_risk_pass1") {
            do {
                try cptPSO_RiskComputePass1 = device.makeComputePipelineState(function: cs)
            } catch {
                os_log("Error creating cs_compute_risk_pass1: %@", log: .default, type: .error, error.localizedDescription)
            }
        }

        if let cs = library?.makeFunction(name: "cs_compute_risk_pass2") {
            do {
                try cptPSO_RiskComputePass2 = device.makeComputePipelineState(function: cs)
            } catch {
                os_log("Error creating cs_compute_risk_pass2: %@", log: .default, type: .error, error.localizedDescription)
            }
        }

        if let cs = library?.makeFunction(name: "cs_compute_risk_pass3") {
            do {
                try cptPSO_RiskComputePass3 = device.makeComputePipelineState(function: cs)
            } catch {
                os_log("Error creating cs_compute_risk_pass3: %@", log: .default, type: .error, error.localizedDescription)
            }
        }

        needsInitialization = false
    }
    
    private func initializeBuffers() {
        guard let device else {
            return
        }
        
        let options: MTLResourceOptions = [.hazardTrackingModeTracked, .storageModePrivate]

        let sizes: [Int] = [
            MemoryLayout<__int32_t>.stride,
            MemoryLayout<Float>.stride * Int(DEVAR_Cnt),
            MemoryLayout<CurState>.stride,
            
            MemoryLayout<Float>.stride * Int(maxGammaKernelLength), // For Gamma Kernel
            MemoryLayout<Float>.stride * Int(maxGammaKernelLength * numAdaptationConditions), // For Energy Buffer
            MemoryLayout<Float>.stride * Int(maxGammaKernelLength * numAdaptationConditions), // For Energy Buffer2
            MemoryLayout<Float>.stride * Int(maxContrastKernelLength * numAdaptationConditions), // For contrast kernel (5 of them for 5 brightness)
            MemoryLayout<Float>.stride * Int(maxContrastKernelLength * numAdaptationConditions), // For contrast buffer (5 of them for 5 brightenss)
            MemoryLayout<Float>.stride * Int(numAdaptationConditions), // For response buffer (5 of them for 5 brightness)
            MemoryLayout<Float>.stride * Int(numAdaptationConditions), // For normalized response buffer (5 of them for 5 brightness)
            MemoryLayout<Float>.stride * Int(numAdaptationConditions * 2) // For results (5 energy result each for one adaptation condition)
        ]

        var alignedSize: size_t = 0
        for i in 0..<sizes.count {
            let sa: MTLSizeAndAlign = device.heapBufferSizeAndAlign(length: sizes[i], options: options)
            alignedSize += (sa.size + sa.align - 1) & ~(sa.align - 1)
        }

        let heapDescriptor = MTLHeapDescriptor()
        heapDescriptor.size = alignedSize
        heapDescriptor.hazardTrackingMode = .tracked
        let heap = device.makeHeap(descriptor: heapDescriptor)

        guard let heap else {
            return
        }
        if let buffer = heap.makeBuffer(length: sizes[0], options: options) {
            bufferFrameLumaSum = buffer
        }
        if let buffer = heap.makeBuffer(length: sizes[1], options: options) {
            bufferData = buffer
        }
        if let buffer = heap.makeBuffer(length: sizes[2], options: options) {
            bufferCurState = buffer
        }
        if let buffer = heap.makeBuffer(length: sizes[3], options: options) {
            bufferGammaKernel = buffer
        }
        if let buffer = heap.makeBuffer(length: sizes[4], options: options) {
            bufferEnergy = buffer
        }
        if let buffer = heap.makeBuffer(length: sizes[5], options: options) {
            bufferEnergy2 = buffer
        }
        if let buffer = heap.makeBuffer(length: sizes[6], options: options) {
            bufferContrastKernel = buffer
        }
        if let buffer = heap.makeBuffer(length: sizes[7], options: options) {
            bufferContrast = buffer
        }
        if let buffer = heap.makeBuffer(length: sizes[8], options: options) {
            bufferResponses = buffer
        }
        if let buffer = heap.makeBuffer(length: sizes[9], options: options) {
            bufferResponsesNorm = buffer
        }
        if let buffer = heap.makeBuffer(length: sizes[10], options: options) {
            bufferResults = buffer
        }
    }
    
    private func computeFrameRateIndexSelection() -> Int {
        let size = (MemoryLayout.stride(ofValue: standardFrameRates) / MemoryLayout.stride(ofValue: standardFrameRates.0))
        var resultIdx = 0
        var minDelta: Float = 1000.0
        
        for i in 0..<size {
            let delta = withUnsafePointer(to: standardFrameRates) { ptr in
                ptr.withMemoryRebound(to: Float.self, capacity: MemoryLayout.stride(ofValue: standardFrameRates)) {
                    return $0[i] - fps
                }
            }

            if delta <= minDelta {
                minDelta = delta
                resultIdx = i
            }
        }
        
        return resultIdx
    }
    
    enum DeviceClass {
        case watch, mac, iphone, ipad
    }

    private func computeDisplaySizeIndexSelection(_ deviceClass: DeviceClass) {
        /*
         The three standard sizes refer to the approximate dimensions of
         Apple Watch, iPhone, and MacBook. The units are degrees of visual angle.
         The actual input to the metric is area, in square degrees. That is then
         matched/approximated by the nearest standard size, and a one-time
         adjustment is also made to deal with the actual size/area not exactly
         matching one of the standards.
         */
        
        switch deviceClass {
        case .watch:
            idxEquivalentKernelIndex = 0
            idxEquivalentSize = 0
            area = 22.5
            break
        case .iphone:
            idxEquivalentKernelIndex = 1
            idxEquivalentSize = 1
            area = 250
            break
        case .ipad:
            idxEquivalentKernelIndex = 2
            idxEquivalentSize = 2
            area = 1000
        case .mac:
            idxEquivalentKernelIndex = 2
            idxEquivalentSize = 3
            area = 1265.625
            break
        }
    }
    
    private func prepareGammaKernel(width: Float, shape: Float, scale: Float) -> ArrayData {
        var array: [Float] = Array(repeating: 0, count: Int(maxGammaKernelLength))
        
        let quantile: Float = 0.99
        let divs: [Float] = [1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0]
        
        let divIdx = Int(shape - 1)
        let div = divs[divIdx]
        
        var t: Float = 0.0
        var length = 0
        
        var sum: Float = 0.0
        
        while sum / width <= quantile || t < 2.0 {
            t += 1.0 / width
            let val = (expf(-(t / scale)) * powf(scale, -shape) * powf(t, (-1.0 + shape))) / div
            sum += val
            array[length] = val
            length = length + 1
        }
        
        return ArrayData(length: UInt(length), array: array)
    }
    
    private func prepareContrastKernels() -> KernelData {
        let size = MemoryLayout<Float>.stride * Int(maxContrastKernelLength) * Int(numAdaptationConditions)
        var array: [Float] = Array(repeating: 0, count: size)
        var length: [UInt] = Array(repeating: 0, count: Int(numAdaptationConditions))

        let kernels = UMFKernels()
        for i in 0..<Int(numAdaptationConditions) {
            var idx = Int(maxContrastKernelLength) * i
            length[i] = 0
            for value in kernels.UMFKernels[idxEquivalentKernelIndex][i][idxFrameRate] {
                array[idx] = Float(value)
                length[i] += 1
                idx += 1
            }
        }

        return KernelData(length: length, array: array)
    }
    
    private func resetState() {
        idxFrameRate = computeFrameRateIndexSelection()
        computeDisplaySizeIndexSelection(.mac)

        // Update GammaKernel
        let gammaKernelData = prepareGammaKernel(width: fps, shape: energyPoolGammaShape, scale: energyPoolGammaScale)
        
        // Reset ContrastBuffers, and update ContrastKernels
        let kernelData = prepareContrastKernels()
        
        // Reset CurState buffer
        let displaySize = withUnsafePointer(to: standardSizes) { ptr in
            ptr.withMemoryRebound(to: Float.self, capacity: MemoryLayout.stride(ofValue: standardSizes)) {
                return $0[idxEquivalentSize]
            }
        }

        var curState = CurState(uContrastBufStartIdx: 0,
                                uGammaKernelLen: UInt32(gammaKernelData.length),
                                fAdaptationLevel: nits * avl,
                                fMuAdapt: 1.0 - exp(-1.0 / (tauAdapt * fps)),
                                fMuMitigation: 1.0 - exp(-1.0 / (tauMitigation * fps)),
                                fResponseAdjust: pow(sqrt(area * 1.6) / displaySize, 2.0 * cA) * (gain / pow(fps, 1.0 / energyPoolExponent)),
                                fPoolEnergy: 0,
                                fPoolEnergy2: 0,
                                fMitigationContrastFactor: 0,
                                fNIU0: 0, fNIU1: 0, fNIU2: 0,
                                fRiskValue: 0,
                                fTime: 0,
                                fEDR: 1,
                                uEnergyStartIdx: 0,
                                uCurStartIdx: (0, 0, 0, 0, 0),
                                uKernelLen: (0, 0, 0, 0, 0),
                                fContrastKernelMagnitude: (0, 0, 0, 0, 0),
                                fNormEnergyThreshold: (0, 0, 0, 0, 0))
        
        for i in 0..<Int(numAdaptationConditions) {

            withUnsafeMutablePointer(to: &curState.uKernelLen) { kernelLengthBuffer in
                kernelLengthBuffer.withMemoryRebound(to: uint.self, capacity: MemoryLayout<uint>.stride * Int(numAdaptationConditions)) {
                    $0[i] = UInt32(kernelData.length[i])
                }
            }

            let startIdx = Int(maxContrastKernelLength) * i
            var sum: Float = 0
            for j in 0..<Int(kernelData.length[i]) {
                sum = sum + pow(kernelData.array[startIdx + j], 2.0)
            }

            withUnsafeMutablePointer(to: &curState.fContrastKernelMagnitude) { magnitude in
                magnitude.withMemoryRebound(to: Float.self, capacity: MemoryLayout<Float>.stride * Int(numAdaptationConditions)) {
                    $0[i] = sum
                }
            }

           let normEnergyThresholdValue = withUnsafePointer(to: NormEnergyThreshold) {
                return withUnsafePointer(to: $0[idxFrameRate]) {
                    return $0.withMemoryRebound(to: Float.self, capacity: MemoryLayout.stride(ofValue: $0)) {
                        return $0[i]
                    }
                }
           }

            withUnsafeMutablePointer(to: &curState.fNormEnergyThreshold) { normEnergy in
                normEnergy.withMemoryRebound(to: Float.self, capacity: MemoryLayout<Float>.stride * Int(numAdaptationConditions)) {
                    $0[i] = normEnergyThresholdValue
                }
            }
        }

        guard let device else {
            return
        }

        let contrastKernelSize = MemoryLayout<Float>.stride * Int(maxContrastKernelLength) * Int(numAdaptationConditions)

        guard let bufferContrastKernels = device.makeBuffer(bytes: kernelData.array, length: contrastKernelSize, options: .storageModeShared) else {
            return
        }
        let gammaKernelSize = MemoryLayout<Float>.stride * Int(gammaKernelData.length)
        guard let gammaBuffer = device.makeBuffer(bytes: gammaKernelData.array, length: gammaKernelSize, options: .storageModeShared) else {
            return
        }

        guard let commandQueue else {
            return
        }

        guard let currentBuffer = device.makeBuffer(bytes: &curState, length: MemoryLayout<CurState>.stride, options: .storageModeShared) else {
            return
        }
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            return
        }

        if let encoder = commandBuffer.makeBlitCommandEncoder() {
            if let buffer = bufferCurState {
                encoder.copy(from: currentBuffer, sourceOffset: 0, to: buffer, destinationOffset: 0, size: MemoryLayout<CurState>.stride)
            }
            if let buffer = bufferGammaKernel {
                encoder.copy(from: gammaBuffer, sourceOffset: 0, to: buffer, destinationOffset: 0, size: gammaKernelSize)
            }
            if let buffer = bufferContrastKernel {
                encoder.copy(from: bufferContrastKernels, sourceOffset: 0, to: buffer, destinationOffset: 0, size: contrastKernelSize)
            }

            if let buffer = bufferContrast {
                encoder.fill(buffer: buffer, range: 0..<contrastKernelSize, value: 0)
            }
            if let buffer = bufferEnergy {
                encoder.fill(buffer: buffer, range: 0..<gammaKernelSize, value: 0)
            }
            if let buffer = bufferEnergy2 {
                encoder.fill(buffer: buffer, range: 0..<gammaKernelSize, value: 0)
            }
            if let buffer = bufferDataDebug {
                encoder.fill(buffer: buffer, range: 0..<MemoryLayout<Float>.stride * Int(DEVAR_Cnt), value: 0)
            }
            encoder.endEncoding()
        }
        commandBuffer.commit()

        if inTestingMode {
            return
        }

        guard let commandBuffer: MTLCommandBuffer = commandQueue.makeCommandBuffer() else {
            return
        }
        
        commandBuffer.commit()
    }
    
    private func processFrame(_ commandBuffer: MTLCommandBuffer, _ sourceSurface: IOSurface, _ outSurface: IOSurface) {

        let width = IOSurfaceGetWidth(sourceSurface)
        let height = IOSurfaceGetHeight(sourceSurface)

        bufferConstants.fPixelCnt = Float(width * height)

        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: pixelFormatForSurface(outSurface), width: width, height: height, mipmapped: false)
        descriptor.usage = .shaderWrite
        processedTexture = device?.makeTexture(descriptor: descriptor, iosurface: outSurface, plane: 0)

        // Clear bufFrameLumaSum
        if let bce = commandBuffer.makeBlitCommandEncoder(), let buffer = bufferFrameLumaSum {
            bce.fill(buffer: buffer, range: 0..<buffer.length, value: 0)
            bce.endEncoding()
        }

        guard let cce = commandBuffer.makeComputeCommandEncoder() else {
            return
        }
        cce.label = "Primary Compute Encoder"
        
        if let pass0 = cptPSO_RiskComputePass0 {
            cce.setComputePipelineState(pass0)
            cce.setTexture(sourceTexture, index: 0)
            cce.setTexture(processedTexture, index: 1)
            cce.setBytes(&bufferConstants, length: MemoryLayout<ConstBuf>.stride, index: 0)
            
            if let buffer = bufferFrameLumaSum {
                cce.setBuffer(buffer, offset: 0, index: Int(BUFIDX_pLumaSum))
            }
            
            if let buffer = bufferData {
                cce.setBuffer(buffer, offset: 0, index: Int(BUFIDX_pDebug))
            }
            
            if let buffer = bufferGammaKernel {
                cce.setBuffer(buffer, offset: 0, index: Int(BUFIDX_pGammaKernel))
            }
            
            if let buffer = bufferCurState {
                cce.setBuffer(buffer, offset: 0, index: Int(BUFIDX_CurState))
            }
            
            let threadWidth = pass0.threadExecutionWidth
            let threadGroup = pass0.maxTotalThreadsPerThreadgroup / threadWidth
            let threadPerThreadgroup = MTLSizeMake(threadWidth, threadGroup, 1)
            let threadsPerGrid = MTLSizeMake(width, height, 1)
            cce.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadPerThreadgroup)
        }

        if let pass1 = cptPSO_RiskComputePass1 {
            cce.setComputePipelineState(pass1)
            cce.setBytes(&bufferConstants, length: MemoryLayout<ConstBuf>.stride, index: Int(BUFIDX_cParams))
            
            if let buffer = bufferFrameLumaSum {
                cce.setBuffer(buffer, offset: 0, index: Int(BUFIDX_pLumaSum))
            }
            
            if let buffer = bufferData {
                cce.setBuffer(buffer, offset: 0, index: Int(BUFIDX_pDebug))
            }
            
            if let buffer = bufferGammaKernel {
                cce.setBuffer(buffer, offset: 0, index: Int(BUFIDX_pGammaKernel))
            }
            
            if let buffer = bufferCurState {
                cce.setBuffer(buffer, offset: 0, index: Int(BUFIDX_CurState))
            }
            
            if let buffer = bufferContrastKernel {
                cce.setBuffer(buffer, offset: 0, index: Int(BUFIDX_pContrastKernels))
            }
            
            if let buffer = bufferContrast {
                cce.setBuffer(buffer, offset: 0, index: Int(BUFIDX_pContrast))
            }
            
            if let buffer = bufferResponses {
                cce.setBuffer(buffer, offset: 0, index: Int(BUFIDX_pResponses))
            }
            
            if let buffer = bufferResponsesNorm {
                cce.setBuffer(buffer, offset: 0, index: Int(BUFIDX_pResponsesNorm))
            }
            
            cce.dispatchThreads(MTLSizeMake(Int(maxContrastKernelLength * numAdaptationConditions), 1, 1), threadsPerThreadgroup: MTLSizeMake(Int(maxContrastKernelLength), 1, 1))
        }
        
        if let pass2 = cptPSO_RiskComputePass2 {
            cce.setComputePipelineState(pass2)
            cce.setBytes(&bufferConstants, length: MemoryLayout<ConstBuf>.stride, index: Int(BUFIDX_cParams))
            
            if let buffer = bufferData {
                cce.setBuffer(buffer, offset: 0, index: Int(BUFIDX_pDebug))
            }
            
            if let buffer = bufferGammaKernel {
                cce.setBuffer(buffer, offset: 0, index: Int(BUFIDX_pGammaKernel))
            }
            
            if let buffer = bufferCurState {
                cce.setBuffer(buffer, offset: 0, index: Int(BUFIDX_CurState))
            }
            
            if let buffer = bufferResponses {
                cce.setBuffer(buffer, offset: 0, index: Int(BUFIDX_pResponses))
            }
            
            if let buffer = bufferResponsesNorm {
                cce.setBuffer(buffer, offset: 0, index: Int(BUFIDX_pResponsesNorm))
            }
            
            if let buffer = bufferEnergy {
                cce.setBuffer(buffer, offset: 0, index: Int(BUFIDX_pEnergy))
            }
            
            if let buffer = bufferEnergy2 {
                cce.setBuffer(buffer, offset: 0, index: Int(BUFIDX_pEnergy2))
            }
            
            if let buffer = bufferResults {
                cce.setBuffer(buffer, offset: 0, index: Int(BUFIDX_pResults))
            }
            
            cce.dispatchThreads(MTLSizeMake(Int(maxGammaKernelLength * numAdaptationConditions), 1, 1), threadsPerThreadgroup: MTLSizeMake(Int(maxGammaKernelLength), 1, 1))
        }
        
        if let pass3 = cptPSO_RiskComputePass3 {
            cce.setComputePipelineState(pass3)
            cce.setBytes(&bufferConstants, length: MemoryLayout<ConstBuf>.stride, index: Int(BUFIDX_cParams))
            
            if let buffer = bufferCurState {
                cce.setBuffer(buffer, offset: 0, index: Int(BUFIDX_CurState))
            }
            
            if let buffer = bufferData {
                cce.setBuffer(buffer, offset: 0, index: Int(BUFIDX_pDebug))
            }
            
            if let buffer = bufferResults {
                cce.setBuffer(buffer, offset: 0, index: Int(BUFIDX_pResults))
            }
            
            cce.dispatchThreads(MTLSizeMake(1, 1, 1), threadsPerThreadgroup: MTLSizeMake(1, 1, 1))
        }
        
        cce.endEncoding()

        // Copy debug info to shared buffer for readback
        if bufferConstants.bDebug && !bufferConstants.bProtected {
            if let bce = commandBuffer.makeBlitCommandEncoder() {
                if let bufferData = bufferData, let bufferDataDebug = bufferDataDebug {
                    bce.copy(from: bufferData, sourceOffset: 0, to: bufferDataDebug, destinationOffset: 0, size: MemoryLayout<Float>.stride * Int(DEVAR_Cnt))
                }
                bce.endEncoding()
            }
        }
    }
    
    private func pixelFormatForSurface(_ surface: IOSurface) -> MTLPixelFormat {
        switch IOSurfaceGetPixelFormat(surface) {
        case kCVPixelFormatType_32RGBA:
            return .rgba8Unorm
        case kCVPixelFormatType_32BGRA:
            return .bgra8Unorm
        case kCVPixelFormatType_64RGBAHalf:
            return .rgba16Float
        case kCVPixelFormatType_30RGBLEPackedWideGamut:
            return .bgr10_xr
        default:
            return .rgba8Unorm
        }
    }
    
    @objc public func processSurface(sourceSurface: IOSurface, timestamp: CFTimeInterval, destinationSurface: IOSurface, options: NSDictionary) {

        if needsInitialization {
            initialize()
        }

        var deltaT = timestamp - previousSurfaceTime
        deltaT = max(deltaT, 1.0 / 120.0)
        deltaT = min(deltaT, 1.0 / 24.0)

        frameDeltas[framePoolIndex] = deltaT
        framePoolIndex = (framePoolIndex + 1) % Int(FPS_Pool_Cnt)

        var maxDelta = 0.001
        var minDelta = 10.0
        var sumDelta = 0.0
        for i in 0..<FPS_Pool_Cnt {
            minDelta = min(minDelta, frameDeltas[Int(i)])
            maxDelta = max(maxDelta, frameDeltas[Int(i)])
            sumDelta += frameDeltas[Int(i)]
        }

        let oldFps = fps
        fps = Float(FPS_Pool_Cnt - 2) / Float(sumDelta - minDelta - maxDelta)
        previousSurfaceTime = timestamp

        let width = IOSurfaceGetWidth(sourceSurface)
        let height = IOSurfaceGetHeight(sourceSurface)

        // The embedded video movie.mp4 uses NTSC color space.
        // Other video formats will need to indicate different uEOTF and uColSpace values
        
        bufferConstants.uEOTF = uint(SRGB)
        bufferConstants.uColSpace = uint(BT709)

        if (timestamp - previousSurfaceTime) > 1.0 || abs(fps - oldFps) >= 5.0 {
            resetState()
        }

        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: pixelFormatForSurface(sourceSurface), width: width, height: height, mipmapped: false)
        
        descriptor.usage = .shaderRead
        guard let device else {
            return
        }
        
        sourceTexture = device.makeTexture(descriptor: descriptor, iosurface: sourceSurface, plane: 0)
        
        bufferConstants.maxNits = (options["displayMaxNits"] as? NSNumber ?? NSNumber(value: 0)).floatValue
        bufferConstants.edr = (options["sourceSurfaceEDR"] as? NSNumber ?? NSNumber(value: 0)).floatValue
        bufferConstants.displayEDR = (options["displayEDRFactor"] as? NSNumber ?? NSNumber(value: 1)).floatValue
        bufferConstants.fps = (options["userFPS"] as? NSNumber ?? NSNumber(value: fps)).floatValue

        guard let commandQueue else {
            return
        }
        
        let semaphore = DispatchSemaphore(value: 0)
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            return
        }
        
        commandBuffer.addCompletedHandler({ [weak self] command in
            semaphore.signal()

            guard let self = self else {
                return
            }

            if self.bufferConstants.bDebug {
                if let validationCallback = self.validationCallback, let ptr = self.bufferDataDebug?.contents() {
                    let apl = ptr.advanced(by: MemoryLayout<Float>.stride * Int(DEVAR_APL)).bindMemory(to: Float.self, capacity: 1).pointee
                    let risk = ptr.advanced(by: MemoryLayout<Float>.stride * Int(DEVAR_Risk)).bindMemory(to: Float.self, capacity: 1).pointee
                    let mitigate = ptr.advanced(by: MemoryLayout<Float>.stride * Int(DEVAR_MitigateCF)).bindMemory(to: Float.self, capacity: 1).pointee

                    validationCallback(apl / 1000.0, risk / 100.0, mitigate)
                }
            }
        })

        processFrame(commandBuffer, sourceSurface, destinationSurface)

        commandBuffer.commit()
        semaphore.wait()
    }
}
