//
//  SharedMetalHeader.h
//  VideoFlashingReduction
//

#ifndef SharedHeader_h
#define SharedHeader_h

// For encoding
#define LINEAR  0
#define SRGB    1
#define REC709  2
#define PQ      3

// For color primaries
#define BT709   0
#define P3      1
#define BT2020  2

// For debug
#define DEVAR_APL           0
#define DEVAR_AdaptLevel    1
#define DEVAR_FrameContrast 2
#define DEVAR_Response      3
#define DEVAR_Response2     4
#define DEVAR_Energypool    5
#define DEVAR_Energypool2   6
#define DEVAR_Risk          7
#define DEVAR_MitigateCF    8
#define DEVAR_EDR           9
#define DEVAR_DisplayEDR    10
#define DEVAR_MuAdapt       11
#define DEVAR_FPS           12
#define DEVAR_VFML          13
#define DEVAR_Cnt           14

#ifdef __METAL_VERSION__
#define NS_ENUM(_type, _name) enum _name : _type _name; enum _name : _type
#define NSInteger metal::int32_t
#define CONST constant
#else
#import <Foundation/Foundation.h>

#define CONST const
#endif

#include <simd/simd.h>

// FPS pool count
static CONST int FPS_Pool_Cnt = 10;

static CONST int maxContrastKernelLength = 32;
static CONST int maxGammaKernelLength = 256;
static CONST int numAdaptationConditions = 5;
static CONST float lumaScaler = 1000.0;

// For GPU Buffer Idx
static CONST int BUFIDX_cParams =        0;
static CONST int BUFIDX_pLumaSum =        1;
static CONST int BUFIDX_pDebug =          2;
static CONST int BUFIDX_pGammaKernel =     3;
static CONST int BUFIDX_CurState =         4;
static CONST int BUFIDX_pContrastKernels = 5;
static CONST int BUFIDX_pContrast =        6;
static CONST int BUFIDX_pResponses =       7;
static CONST int BUFIDX_pResponsesNorm =   8;
static CONST int BUFIDX_pEnergy =          9;
static CONST int BUFIDX_pEnergy2 =         10;
static CONST int BUFIDX_pResults =         11;

// Energy Threshold
static CONST float NormEnergyThreshold[7][5] = {
    {1.400000, 1.300000, 1.300000, 1.250000, 1.250000},
    {1.400000, 1.300000, 1.300000, 1.250000, 1.250000},
    {1.500000, 1.400000, 1.300000, 1.300000, 1.300000},
    {1.900000, 1.800000, 1.400000, 1.400000, 1.400000},
    {2.250000, 2.100000, 1.500000, 1.500000, 1.500000},
    {2.500000, 2.400000, 2.200000, 2.000000, 2.000000},
    {3.000000, 2.600000, 2.250000, 2.000000, 2.000000}
};

// Define standard sizes, nits, rates

// Apple Watch / iPhone / iPad / macOS
static CONST float standardSizes[] = {6.0, 20.0, 40.0, 45.0};
static CONST float standardNits[] = {0.2, 1, 10, 150, 500};
static CONST float logStandardNits[] = {-0.6990, 0.0, 1.0, 2.1761, 2.6990}; // log10(standardNits)
static CONST float standardFrameRates[] = {24.0, 25.0, 30.0, 50.0, 60.0, 90.0, 120.0};

typedef enum : int32_t {
    FlashingLightsColorSpaceUnknown = 0,
    FlashingLightsColorSpaceDisplayP3 = 1, // The Display P3 color space, created by Apple.
    
    FlashingLightsColorSpaceDisplayP3_HLG = 2, // The Display P3 color space, using the HLG transfer function.
    // FlashingLightsColorSpaceDisplayP3_PQ_EOTF = 3, // DEPRECATED The Display P3 color space, using the PQ transfer function.
    FlashingLightsColorSpaceExtendedLinearDisplayP3 = 4, // The Display P3 color space with a linear transfer function and extended-range values.
    FlashingLightsColorSpaceSRGB = 5, // The standard Red Green Blue (sRGB) color space.
    FlashingLightsColorSpaceLinearSRGB = 6, // The sRGB color space with a linear transfer function.
    FlashingLightsColorSpaceExtendedSRGB = 7, // The extended sRGB color space.
    FlashingLightsColorSpaceExtendedLinearSRGB = 8, // The sRGB color space with a linear transfer function and extended-range values.
    FlashingLightsColorSpaceGenericGrayGamma2_2 = 9, // The generic gray color space that has an exponential transfer function with a power of 2.2.
    FlashingLightsColorSpaceLinearGray = 10, // The gray color space using a linear transfer function.
    FlashingLightsColorSpaceExtendedGray = 11, // The extended gray color space.
    FlashingLightsColorSpaceExtendedLinearGray = 12, // The extended gray color space with a linear transfer function.
    FlashingLightsColorSpaceGenericRGBLinear = 13, // The generic RGB color space with a linear transfer function.
    FlashingLightsColorSpaceGenericCMYK = 14, //The generic CMYK color space.
    FlashingLightsColorSpaceGenericXYZ = 15, // The XYZ color space, as defined by the CIE 1931 standard.
    FlashingLightsColorSpaceGenericLab = 16, // The generic LAB color space.
    FlashingLightsColorSpaceACESCGLinear = 17, // The ACEScg color space.
    FlashingLightsColorSpaceAdobeRGB1998 = 18, // The Adobe RGB (1998) color space.
    FlashingLightsColorSpaceDCIP3 = 19, // The DCI P3 color space, which is the digital cinema standard.
    FlashingLightsColorSpaceITUR_709 = 20, // The recommendation of the International Telecommunication Union (ITU) Radiocommunication sector for the BT.709 color space.
    FlashingLightsColorSpaceROMMRGB = 21, // The Reference Output Medium Metric (ROMM) RGB color space.
    FlashingLightsColorSpaceITUR_2020 = 22, // The recommendation of the International Telecommunication Union (ITU) Radiocommunication sector for the BT.2020 color space.
    // FlashingLightsColorSpaceITUR_2020_HLG = 23, // DEPRECATED The recommendation of the International Telecommunication Union (ITU) Radiocommunication sector for the BT.2020 color space, with the HLG transfer function.
    FlashingLightsColorSpaceITUR_2020_PQ_EOTF = 24, // The recommendation of the International Telecommunication Union (ITU) Radiocommunication sector for the BT.2020 color space, with the PQ transfer function.
    FlashingLightsColorSpaceExtendedLinearITUR_2020 = 25, // The recommendation of the International Telecommunication Union (ITU) Radiocommunication sector for the BT.2020 color space, with a linear transfer function and extended range values.
    FlashingLightsColorSpaceGenericRGB = 26, // The name of the generic RGB color space.
    FlashingLightsColorSpaceGenericGray = 27, // The name of the generic gray color space.
    FlashingLightsColorSpaceITUR_2100_PQ = 28,
    FlashingLightsColorSpaceDisplayP3_PQ = 29,
    FlashingLightsColorSpaceNTSC = 30,
    FlashingLightsColorSpacePAL = 31,
    FlashingLightsColorSpaceITUR_2100_HLG = 32,
} FlashingLightsColorSpace;

struct ConstBuf {
    float fPixelCnt;
    bool bMitigate;
    bool bDebug;
    bool bCopyOnly;
    
    bool bSWGammaCorrection;
    bool bProtected;
    uint uEOTF;
    uint uColSpace;
    
    float maxNits;
    float edr;
    float displayEDR;
    float fps;
};

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpadded"
struct CurState {
    uint uContrastBufStartIdx;
    uint uGammaKernelLen;
    float fAdaptationLevel;
    float fMuAdapt;
    float fMuMitigation;
    
    float fResponseAdjust;
    float fEnergypoolExponent;
    float fPoolEnergy;
    float fPoolEnergy2;
    
    float fMitigationContrastFactor;
    float fNIU0;
    float fNIU1;
    float fNIU2;
    
    float fRiskValue;
    float fTime;
    float fEDR;
    uint uEnergyStartIdx;
    
    uint uCurStartIdx[numAdaptationConditions];
    uint uKernelLen[numAdaptationConditions];
    float fContrastKernelMagnitude[numAdaptationConditions];
    float fNormEnergyThreshold[numAdaptationConditions];
};

#pragma clang diagnostic pop

#endif /* SharedHeader_h */
