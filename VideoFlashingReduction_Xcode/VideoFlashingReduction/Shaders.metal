//
//  Shaders.metal
//  VideoFlashingReduction
//

#include <metal_stdlib>
#include "SharedMetalHeader.h"

using namespace metal;

typedef struct
{
    float4 f4Pos [[position]];
    float2 f2UV;
} ColorInOut;

//====================================================================
// Utility function for encoding conversion
//====================================================================
constant float fAlpha = 1.0993;
constant float fBeta = 0.0181;
template <class T>
inline T Rec709ToLinear(T fCol) {
    T fTemp = pow((fCol + fAlpha - 1.0f)/fAlpha, 1.0f/0.45f) - fCol/4.5f;
    return fTemp * step(4.5f * fBeta, fCol) + fCol/4.5f;
}

template <class T>
inline T LinearToRec709(T fCol) {
    T fTemp = (fAlpha * pow(fCol, 0.45) - fAlpha - 1.0f - 4.5 * fCol);
    return fTemp * step(fBeta, fCol) + 4.5f * fCol;
}

template <class T>
inline T SRGBToLinear(T fCol){
    T fTemp = pow((fCol + 0.055f)/1.055f, 2.4f) - fCol / 12.92f;
    return fTemp * step(0.04045, fCol) + fCol / 12.92;
}

template <class T>
inline T LinearToSRGB (T fCol){
    T fTemp = 1.055f * pow(fCol, 1.f/2.4f) - fCol * 12.92 - 0.055;
    return fTemp * step(0.0031308, fCol) + fCol * 12.92;
}

constant float fM1 = 2610.0 / 4096.0 / 4.0;
constant float fM2 = 2523.0 / 4096.0 * 128.0;
constant float fC1 = 3424.0 / 4096.0;
constant float fC2 = 2413.0 * 32.0 / 4096.0;
constant float fC3 = 2392.0 / 4096.0 * 32.0;

// Range for fL: [0, 10000] for 0 to 10000 nits
template <class T>
inline T LinearToPQ(T fL) {
    T fx = pow(fL / 10000.f, fM1);
    fx = (fC1 + fC2 * fx) / (1.f + fC3 * fx);
    return pow(fx, fM2);
}

// Range for fPQ: [0, 1]
template <class T>
inline T PQToLinear(T fPQ) {
    T fx = pow(fPQ, 1.f / fM2);
    fx = (fx - fC1) / (fC2 - fC3 * fx);
    fx = max(fx, 0.f);
    return 10000.f * pow(fx, 1.f / fM1);
}

//====================================================================
// Utility function for color space conversion
//====================================================================
inline float3 BT709_2_XYZ(float3 f3RGB) {
    const float3x3 mTran = float3x3(0.4124564,  0.3575761,  0.1804375,
                                    0.2126729,  0.7151522,  0.0721750,
                                    0.0193339,  0.1191920,  0.9503041);
    return f3RGB * mTran;
}

inline float3 XYZ_2_BT709(float3 f3XYZ) {
    const float3x3 mTran = float3x3(+3.241003232976359, -1.537398969488786, -0.498615881996363,
                                    -0.969224252202516, +1.875929983695176, +0.041554226340085,
                                    +0.055639419851975, -0.204011206123910, +1.057148977187533);
    return f3XYZ * mTran;
}

inline float3 P3D65RGB_2_XYZ(float3 f3RGB) {
    const float3x3 mTran = float3x3(+0.486570948648216, +0.265667693169093, +0.198217285234362,
                                    +0.228974564069749, +0.691738521836506, +0.079286914093745,
                                    +0.000000000000000, +0.045113381858903, +1.043944368900976);
    return f3RGB * mTran;
}

inline float3 XYZ_2_P3(float3 f3XYZ) {
    const float3x3 mTran = float3x3(+2.493509123934610, -0.931388179404779, -0.402712756741652,
                                    -0.829473213929554, +1.762630579600303, +0.02364237105589,
                                    +0.035851264433918, -0.076183936922076, +0.957029586694311);
    return f3XYZ * mTran;
}

inline float3 BT2020D65RGB_2_XYZ(float3 f3RGB) {
    const float3x3 mTran = float3x3(+0.636958048301291, +0.144616903586208, +0.168880975164172,
                                    +0.262700212011267, +0.677998071518871, +0.059301716469862,
                                    +0.000000000000000, +0.028072693049087, +1.060985057710791);
    return f3RGB * mTran;
}

inline float3 XYZ_2_BT2020D65RGB(float3 f3XYZ) {
    const float3x3 mTran = float3x3(+1.716651187971268, -0.355670783776392, -0.253366281373660,
                                    -0.666684351832489, +1.616481236634939, +0.015768545813911,
                                    +0.017639857445311, -0.042770613257808, +0.942103121235474);
    return f3XYZ * mTran;
}

//====================================================================
// Utility functions
//====================================================================
template <class T>
inline T linearSample(float x, device T* pData, int iLen) {
    x = saturate(x) * float(iLen);
    T a = pData[int(floor(x))];
    T b = pData[int(ceil(x))];
    return mix(a, b, fract(x));
}

inline float tonemap(float fLumaIn, float fPeakConstrain) {
    // Define headroom
    const float3 DR_in = float3(0, 1, 0.03623); // [blacklevel, whitepoint, LumAdaptation]
    const float3 DR_out = float3(0, fPeakConstrain, 0.03623); // [blacklevel, whitepoint, LumAdaptation]
    // define some constants
    const float pb1 = DR_in.x/(DR_in.z+DR_in.x); // Zero
    const float pw1 = DR_in.y/(DR_in.z+DR_in.y); // 100/3.1623 ~ 30
    const float pb2 = DR_out.x/(DR_out.z+DR_out.x);
    const float pw2 = DR_out.y/(DR_out.z+DR_out.y);
    const float k1 = pb2 - pb1*(pw2-pb2)/(pw1-pb1);
    const float k2 = k1 + (pw2-pb2)/(pw1-pb1);
    const float u1k1 = DR_in.z * k1;
    const float c1 = DR_out.z/DR_out.y * u1k1;
    const float c2 = DR_out.z*DR_in.y*DR_out.y * k2;
    const float c3 = DR_in.z - u1k1;
    const float c4 = -DR_in.y*(k2-1.0);
    return fPeakConstrain*(c1+c2*fLumaIn)/(c3+c4*fLumaIn);
}

inline float3 linearize(float3 f3Col, constant ConstBuf& cParams) {
    switch (cParams.uEOTF) {
        case SRGB:      return SRGBToLinear(f3Col);
        case REC709:    return Rec709ToLinear(f3Col);
        case PQ:        return PQToLinear(f3Col) / cParams.maxNits;
        default:        return f3Col;
    }
}

inline float3 encode(float3 f3Col, constant ConstBuf& cParams) {
    switch (cParams.uEOTF) {
        case SRGB:      return LinearToSRGB(f3Col);
        case REC709:    return LinearToRec709(f3Col);
        case PQ:        return LinearToPQ(f3Col * cParams.maxNits);
        default:        return f3Col;
    }
}

inline float getLuma(float3 f3Col, constant ConstBuf& cParams) {
    switch (cParams.uColSpace) {
        case BT709:     return BT709_2_XYZ(f3Col).y;
        case P3:        return P3D65RGB_2_XYZ(f3Col).y;
        case BT2020:    return BT2020D65RGB_2_XYZ(f3Col).y;
        default:        return max(f3Col.r, max(f3Col.g, f3Col.b));
    }
}

kernel void
cs_compute_risk_pass0(texture2d<float, access::read> texVideoFrame [[texture(0)]],
                      texture2d<float, access::write> texProcFrame [[texture(1)]],
                      constant ConstBuf& cParams [[buffer(BUFIDX_cParams)]],
                      device atomic_uint* apLumaSum [[buffer(BUFIDX_pLumaSum)]],
                      device float* pDebug [[buffer(BUFIDX_pDebug)]],
                      device float* pGammaKernel [[buffer(BUFIDX_pGammaKernel)]],
                      device CurState& curState [[buffer(BUFIDX_CurState)]],
                      uint uTGid [[thread_index_in_threadgroup]],
                      uint2 u2Gid [[thread_position_in_grid]],
                      uint2 u2GSize [[threads_per_grid]],
                      uint2 u2Offset [[grid_size]])
{
    if ( cParams.bCopyOnly )
    {
        float4 f4Col = texVideoFrame.read(u2Gid);
        texProcFrame.write(f4Col, u2Gid);
    }
    else
    {
        // Luma sum in the threadgroup
        threadgroup atomic_uint tgauLumaSum;
        if (uTGid == 0) {
            atomic_store_explicit(&tgauLumaSum, 0, memory_order_relaxed);
        }
        // Read in video pixel rgb and convert to linear
        // (iOS is already configured with a tex format which will do gamma conversion during load)
        float3 f3Col_orig = texVideoFrame.read(u2Gid).rgb;
        float3 f3Col = linearize(f3Col_orig, cParams) * cParams.displayEDR / cParams.edr;
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Currently to match MATLAB version, luma conversion is modified to align with MATLAB code
        float fLumIn = getLuma(f3Col.rgb, cParams);
        //float fLumIn = dot(f3Col, float3(0.299, 0.587, 0.114));
        
        if (u2Gid.x < texVideoFrame.get_width() && u2Gid.y < texVideoFrame.get_height()){
            atomic_fetch_add_explicit(&tgauLumaSum, uint(fLumIn * lumaScaler), memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (uTGid == 0) {
            uint uPartialSum = atomic_load_explicit(&tgauLumaSum, memory_order_relaxed);
            atomic_fetch_add_explicit(apLumaSum, uPartialSum, memory_order_relaxed);

        }

        float4 f4ColOut = float4(f3Col, 1.0);
        
        // Reduce contrast as a function of fMitigationContrastFactor
        float mContrastScale = ( 1 - curState.fMitigationContrastFactor );
        float mLumfactor = ( 1 - curState.fMitigationContrastFactor );
        float arlum = curState.fAdaptationLevel/lumaScaler;
        
        float3 f3TMed = mContrastScale*(mLumfactor*(f4ColOut.rgb - arlum)) + mLumfactor*arlum;
        f4ColOut.rgb = (cParams.bDebug && u2Gid.x < texVideoFrame.get_width()/2)
        ? f3Col_orig
        : encode(f3TMed * cParams.edr / cParams.displayEDR, cParams);
        
        
        if (cParams.bDebug) {
            // Visual checking on PoolEnergy
            float2 f2xy = float2(u2Gid) / float2(u2GSize);
            f2xy.y = 1.0 - f2xy.y;
            float y = curState.fRiskValue / 100.0;
            if (abs(y - f2xy.y) < 0.005)
                f4ColOut = float4(0.0, cParams.edr, 0.0, 1.0);
            
            // Protected content indicator
            if (all(f2xy < 0.01)) {
                f4ColOut = cParams.bProtected ? float4(cParams.edr,0,0,1) : float4(0,cParams.edr,0,1);
            }
        }
        
        texProcFrame.write(f4ColOut, u2Gid);
    }
}

kernel void
cs_compute_risk_pass1(constant ConstBuf& cParams [[buffer(BUFIDX_cParams)]],
                      device const uint* pLumaSum [[buffer(BUFIDX_pLumaSum)]],
                      device float* pDebug [[buffer(BUFIDX_pDebug)]],
                      device float* pGammaKernel [[buffer(BUFIDX_pGammaKernel)]],
                      device CurState& curState [[buffer(BUFIDX_CurState)]],
                      device const float* pContrastKernels [[buffer(BUFIDX_pContrastKernels)]],
                      device float* pContrast [[buffer(BUFIDX_pContrast)]],
                      device float* pResponses [[buffer(BUFIDX_pResponses)]],
                      device float* pResponsesNorm [[buffer(BUFIDX_pResponsesNorm)]],
                      uint uTGid [[threadgroup_position_in_grid]], // StandardNitLevel_position [0...4]
                      uint uTidInTG [[thread_position_in_threadgroup]], // Position in kernel
                      uint uTid [[thread_position_in_grid]])
{
    const float fAPL = float(*pLumaSum) / lumaScaler / cParams.fPixelCnt * cParams.maxNits;
    
    float fPrevAdaptLevel = curState.fAdaptationLevel;
    if (curState.fTime < 0.0001){ // First frame
        fPrevAdaptLevel = fAPL;
    }
    
    float fCurAdaptLevel = curState.fMuAdapt * fAPL + (1.0 - curState.fMuAdapt) * fPrevAdaptLevel;
    curState.fAdaptationLevel = fCurAdaptLevel;
    
    float fFrameContrast = (fAPL - fCurAdaptLevel) / fCurAdaptLevel;
    
    // For ContrastKernel convolution
    threadgroup float tgfTemp[maxContrastKernelLength];
    threadgroup float tgfTemp1[maxContrastKernelLength];
    threadgroup float tgfTemp2[maxContrastKernelLength];
    
    uint uLen = curState.uKernelLen[uTGid]; // Length of current kernel
    uint uIdxStart = (curState.uCurStartIdx[uTGid] + 1) % uLen;
    uint uBufferIdx = uTidInTG;
    uint uKernelIdx = uTidInTG;
    uint uKernelRevIdx = uTidInTG;
    
    if (uTidInTG < uLen) { // For kernel position [0 ... end]
        uBufferIdx = (uIdxStart + uBufferIdx) % uLen;
        uKernelIdx = uLen - 1 - uKernelIdx; // Flip the kernel to accommodate buffer
    }
    
    uBufferIdx = maxContrastKernelLength * uTGid + uBufferIdx;
    uKernelIdx = maxContrastKernelLength * uTGid + uKernelIdx;
    uKernelRevIdx = maxContrastKernelLength * uTGid + uKernelRevIdx;
    
    float fContrastKernelElement = pContrastKernels[uKernelIdx];
    float fContrastKernelRevElement = pContrastKernels[uKernelRevIdx];
    float fContrastBufferElement = pContrast[uBufferIdx];
    
    // The thread at the end of each buffer updates the buffer and updates startIdx
    if (uTidInTG == (uLen - 1)) {
        fContrastBufferElement = fFrameContrast;
        curState.uCurStartIdx[uTGid] = uIdxStart;
        pContrast[uBufferIdx] = fFrameContrast;
    }
    
    // Convolution
    tgfTemp[uTidInTG] = fContrastKernelElement * fContrastBufferElement;
    tgfTemp1[uTidInTG] = fContrastBufferElement * fContrastBufferElement;
    tgfTemp2[uTidInTG] = fContrastKernelRevElement * fContrastBufferElement;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float fResponse = 0.0;
    float fResponse2 = 0.0;
    float fContrastBufferMagnitude = 0.0;
    if (uTidInTG == 0) {
        for (uint i = 0; i < uLen; ++i) {
            fResponse += tgfTemp[i];
            fResponse2 += tgfTemp2[i];
            fContrastBufferMagnitude += tgfTemp1[i];
        }
        
        pResponses[uTGid] = fResponse;
        pResponsesNorm[uTGid] = fResponse2 / sqrt(fContrastBufferMagnitude * curState.fContrastKernelMagnitude[uTGid] + 0.00001);
    }
    
    if (cParams.bDebug && uTid == 0) {
        pDebug[DEVAR_APL] = fAPL;
        pDebug[DEVAR_AdaptLevel] = fCurAdaptLevel;
        pDebug[DEVAR_FrameContrast] = fFrameContrast;
        pDebug[DEVAR_Response] = pResponses[2];
        pDebug[DEVAR_Response2] = pResponsesNorm[2];
        pDebug[DEVAR_EDR] = cParams.edr;
        pDebug[DEVAR_DisplayEDR] = cParams.displayEDR;
        pDebug[DEVAR_MuAdapt] = curState.fMuAdapt;
        pDebug[DEVAR_FPS] = cParams.fps;
    }
}

kernel void
cs_compute_risk_pass2(constant ConstBuf& cParams [[buffer(BUFIDX_cParams)]],
                      device float* pDebug [[buffer(BUFIDX_pDebug)]],
                      device float* pGammaKernel [[buffer(BUFIDX_pGammaKernel)]],
                      device CurState& curState [[buffer(BUFIDX_CurState)]],
                      device float* pResponses [[buffer(BUFIDX_pResponses)]],
                      device float* pResponsesNorm [[buffer(BUFIDX_pResponsesNorm)]],
                      device float* pEnergy [[buffer(BUFIDX_pEnergy)]],
                      device float* pEnergy2 [[buffer(BUFIDX_pEnergy2)]],
                      device float* pResults [[buffer(BUFIDX_pResults)]],
                      uint uTGid [[threadgroup_position_in_grid]], // StandardNitLevel_position [0...4]
                      uint uTidInTG [[thread_position_in_threadgroup]], // Position in buffer
                      uint uTid [[thread_position_in_grid]])
{
    threadgroup float fTemp[maxGammaKernelLength];
    threadgroup float fTemp2[maxGammaKernelLength];
    
    const uint uLen = curState.uGammaKernelLen;
    const uint uIdxStart = (curState.uEnergyStartIdx + 1) % uLen;
    const uint uEnergyBufBaseIdx = maxGammaKernelLength * uTGid;
    uint uIdx = uTidInTG;
    
    fTemp[uTidInTG] = 0;
    fTemp2[uTidInTG] = 0;
    
    if (uTidInTG < uLen) {
        uIdx = (uIdxStart + uIdx) % uLen;
        uIdx += uEnergyBufBaseIdx;
        fTemp[uTidInTG] = pEnergy[uIdx] * pGammaKernel[uLen - 1 - uTidInTG];
        fTemp2[uTidInTG] = pEnergy2[uIdx] * pGammaKernel[uLen - 1 - uTidInTG];
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // The thread at the end of each buffer updates the buffer and updates startIdx
    if (uTidInTG == (uLen - 1)) {
        float fEnergy = pow(pResponses[uTGid], curState.fEnergypoolExponent);
        float fEnergy2 = pow(pResponsesNorm[uTGid], curState.fEnergypoolExponent);
        
        // Update the most recent Energy
        pEnergy[uIdx] = fEnergy;
        pEnergy2[uIdx] = fEnergy2;
        curState.uEnergyStartIdx = uIdxStart;
        
        fEnergy *= pGammaKernel[0];
        fEnergy2 *= pGammaKernel[0];
        for (uint i = 0; i < (uLen - 1); ++i) { // Last element is the most recent, so skip it since the sum is initialized with that
            fEnergy += fTemp[i];
            fEnergy2 += fTemp2[i];
        }
        
        pResults[uTGid * 2] = pow(fEnergy, 1.0 / curState.fEnergypoolExponent);
        pResults[uTGid * 2 + 1] = pow(fEnergy2, 1.0 / curState.fEnergypoolExponent);
    }
}

kernel void
cs_compute_risk_pass3(constant ConstBuf& cParams [[buffer(BUFIDX_cParams)]],
                      device CurState& curState [[buffer(BUFIDX_CurState)]],
                      device float* pDebug [[buffer(BUFIDX_pDebug)]],
                      device float* pResults [[buffer(BUFIDX_pResults)]])
{
    float fLog10AdaptLevel = log10(curState.fAdaptationLevel);
    float fEnergy, fEnergy2;
    if (fLog10AdaptLevel < logStandardNits[0]){
        fEnergy = pResults[0];
        fEnergy2 = pResults[1];
    } else {
        bool bFound = false;
        for (uint u = 1; u <numAdaptationConditions; ++u) {
            if (fLog10AdaptLevel < logStandardNits[u]) {
                float a = (fLog10AdaptLevel - logStandardNits[u-1])/ (logStandardNits[u] - logStandardNits[u-1]);
                fEnergy = mix(pResults[2 * (u-1)], pResults[2 * u], a);
                fEnergy2= mix(pResults[2 * (u-1) + 1], pResults[2 * u + 1], a);
                bFound = true;
                break;
            }
        }
        if (!bFound) {
            fEnergy = pResults[2 * (numAdaptationConditions - 1)];
            fEnergy2 = pResults[2 * (numAdaptationConditions - 1) + 1];
        }
    }
    fEnergy = fEnergy * curState.fResponseAdjust;
    fEnergy2 = pResults[9];

    curState.fPoolEnergy = fEnergy;
        
    float offset = 33;
    float scale = 200;
    float shape = 3;
    float VFML = 0;
    
    float fRisk = 0;
    if (fEnergy > offset) {
        VFML = 100.0*(1.0 - exp(  -pow( (-offset + fEnergy )/scale ,shape) ) );
    }
    
    uint uIdx = 0;
    float mindelta = 10000.0;
    const float fLogAdaptLevel = log10(curState.fAdaptationLevel + 0.00001);
    for (uint i = 0; i < numAdaptationConditions; ++i) {
        float delta = abs(logStandardNits[i] - fLogAdaptLevel);
        if (delta <= mindelta) {
            mindelta = delta;
            uIdx = i;
        }
    }
    
    float VFML2 = 0;
    if (fEnergy2 > 1.8)
        VFML2 = VFML;
    fRisk = VFML2;
    
    // Calculate contrast mitigation factor
    float gain = 0.5;
    float mitigationThreshold = 0.0;
    float mitigationStrength = 0.0;
    if (fRisk > mitigationThreshold)
        mitigationStrength = saturate(log10(fRisk) * gain);
    
    // Add additional slow down ramping of mitigationStrength
    if (mitigationStrength < curState.fMitigationContrastFactor) {
        mitigationStrength = mitigationStrength*curState.fMuMitigation + curState.fMitigationContrastFactor*(1 - curState.fMuMitigation);
        mitigationStrength = mitigationStrength < 0.01 ? 0.0 : mitigationStrength;
    }
    
    curState.fRiskValue = fRisk;
    curState.fMitigationContrastFactor = mitigationStrength;
    curState.fTime = curState.fTime + 1.0/ cParams.fps;
    
    if (cParams.bDebug) {
        pDebug[DEVAR_Energypool] = fEnergy;
        pDebug[DEVAR_Energypool2] = fEnergy2;
        pDebug[DEVAR_Risk] = fRisk;
        pDebug[DEVAR_MitigateCF] = mitigationStrength;
        pDebug[DEVAR_VFML] = VFML;
    }
}
