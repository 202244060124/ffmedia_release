#ifndef __TYPE_H__
#define __TYPE_H__

/*
 * Decode type support
 */
enum DecodeType {
    DECODE_TYPE_H264 = 0,
    DECODE_TYPE_H265,
    DECODE_TYPE_MJPEG,
    DECODE_TYPE_MAX,
};

/*
 * Encode type support
 */
enum EncodeType {
    ENCODE_TYPE_H264 = 0,
    ENCODE_TYPE_H265,
    ENCODE_TYPE_MJPEG,
    ENCODE_TYPE_MAX,
};

/*
 * RcMode - rate control mode
 * 0 - cbr mode, Constant bit rate
 * 1 - vbr mode, variable bit rate
 */
enum EncodeRcMode {
    ENCODE_RC_MODE_CBR = 0,
    ENCODE_RC_MODE_VBR,
    ENCODE_RC_MODE_FIXQP,
    ENCODE_RC_MODE_AVBR,
};

/*
 * H.264 profile_idc parameter
 * 66  - Baseline profile
 * 77  - Main profile
 * 100 - High profile
 */
enum EncodeProfile {
    ENCODE_PROFILE_BASELINE = 0,
    ENCODE_PROFILE_MAIN,
    ENCODE_PROFILE_HIGH,
};

/*
 * Quality - quality parameter
 * mpp does not give the direct parameter in different protocol.
 * mpp provide total 5 quality level 1 ~ 5
 * 0 - worst
 * 1 - worse
 * 2 - medium
 * 3 - better
 * 4 - best
 */
enum EncodeQuality {
    ENCODE_QUALITY_WORST = 0,
    ENCODE_QUALITY_WORSE,
    ENCODE_QUALITY_MEDIUM,
    ENCODE_QUALITY_BETTER,
    ENCODE_QUALITY_BEST,
};

enum RgaRotate {
    RGA_ROTATE_NONE = 0,
    RGA_ROTATE_90,
    RGA_ROTATE_180,
    RGA_ROTATE_270,
    RGA_ROTATE_VFLIP,  // Vertical Mirror
    RGA_ROTATE_HFLIP,  // Horizontal Mirror
};

enum yuv2RgbMode {
    RGB_TO_RGB = 0,
    YUV_TO_YUV = 0,
    YUV_TO_RGB = 0x1 << 0,
    RGB_TO_YUV = 0x2 << 4,
};

enum SampleFormat {
    SAMPLE_FMT_NONE = -1,
    SAMPLE_FMT_U8,
    SAMPLE_FMT_S16,
    SAMPLE_FMT_S32,
    SAMPLE_FMT_FLT,
    SAMPLE_FMT_U8P,
    SAMPLE_FMT_S16P,
    SAMPLE_FMT_S32P,
    SAMPLE_FMT_FLTP,
    SAMPLE_FMT_G711A,
    SAMPLE_FMT_G711U,
    SAMPLE_FMT_NB
};

enum MEDIA_BUFFER_TYPE {
    BUFFER_TYPE_VIDEO,
    BUFFER_TYPE_AUDIO,
    BUFFER_TYPE_ETC
};

enum SynchronizeType {
    SYNCHRONIZETYPE_AUDIO,
    SYNCHRONIZETYPE_VIDEO,
    SYNCHRONIZETYPE_ABSOLUTE
};

#endif
