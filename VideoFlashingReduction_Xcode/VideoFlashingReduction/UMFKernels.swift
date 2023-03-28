//
//  UMFKernels.swift
//  VideoFlashingReduction
//

import Foundation
import os
@_implementationOnly import VideoFlashingReduction_Bridge_Internal

class UMFKernels {

    private let maxIndex1 = 3
    private let maxIndex2 = 5
    private let maxIndex3 = 7
    private let maxIndex4 = 10
    public var UMFKernels: [[[[Double]]]]

    init() {
        UMFKernels = [[[[Double]]]](repeating: [[[Double]]](repeating: [[Double]](repeating: [Double](repeating: 0, count: maxIndex4), count: maxIndex3), count: maxIndex2), count: maxIndex1)
        initUMFKernels()
    }

    func initUMFKernels() {
        guard let filePath = Bundle(for: VideoProcessor.self).path(forResource: "UMFKernel-Data", ofType: "txt") else {
            return
        }

        var stringData: NSString?
        do {
            stringData = try NSString(contentsOfFile: filePath, encoding: NSASCIIStringEncoding)
        } catch {
            print("Error getting file: %@", error.localizedDescription)
        }
        
        guard let stringData = stringData else {
            return
        }

        let array = stringData.components(separatedBy: "\n")
        var index1 = 0, index2 = 0, index3 = 0
        for line in array {
            guard let lineString = line as String? else {
                continue
            }

            UMFKernels[index1][index2][index3] = lineString.components(separatedBy: ",").map({ return Double($0.trimmingCharacters(in: .whitespacesAndNewlines)) ?? 0 })

            index3 = index3 + 1
            if index3 >= maxIndex3 {
                index3 = 0
                index2 = index2 + 1
            }
            if index2 >= maxIndex2 {
                index2 = 0
                index1 = index1 + 1
            }
            if index1 >= maxIndex1 {
                index1 = 0
            }
        }
    }
}
