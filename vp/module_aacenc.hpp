#ifndef __MODULE__AACENC_HPP__
#define __MODULE__AACENC_HPP__

#include "module/module_media.hpp"
#include "base/ff_type.hpp"

struct AACENCODER;

class ModuleAacEnc : public ModuleMedia
{
    AACENCODER* enc;
    SampleFormat fmt;
    int sample_rate;
    int nb_channels;
    int aot;
    int bit_rate;
    int afterburner;
    int eld_sbr;
    int vbr;

public:
    /*
     * SampleFormat:
     *	SAMPLE_FMT_S16, SAMPLE_FMT_NONE
     * _sample_rate:
     *	96000, 88200, 64000, 48000, 44100, 32000,
     *	24000, 22050, 16000, 12000, 11025, 8000, 0
     * _nb_channels:
     *	1 ~ 8
     */
    ModuleAacEnc(SampleFormat _fmt, int _sample_rate, int _nb_channels);
    ~ModuleAacEnc();
    int init();

    // aot == 2;  "LC"
    // aot == 5;  "HE-AAC"
    // aot == 29; "HE-AACv2"
    // aot == 23; "LD"
    // aot == 39; "ELD"
    void setAot(int _aot) { aot = _aot; }
    int getAot() { return aot; }
    void setBitrate(int bitrate) { bit_rate = bitrate; }
    int getBitrate() { return bit_rate; }
    void setAfterburner(int _afterburner) { afterburner = _afterburner; }
    int getAfterburner() { return afterburner; }
    void setEldSbr(int _eld_sbr) { eld_sbr = _eld_sbr; }
    int getEldSbr() { return eld_sbr; }
    void setVbr(int _vbr) { vbr = _vbr; }
    int gerVbr() { return vbr; }

protected:
    virtual EnQueueResult doEnQueue(MediaBuffer* input_buffer, MediaBuffer* output_buffer) override;

private:
    void close();
};

#endif
