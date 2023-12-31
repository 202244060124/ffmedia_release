#ifndef __FF_SYNCHRONIZE_HPP__
#define __FF_SYNCHRONIZE_HPP__

#include <sys/time.h>
#include <inttypes.h>
#include <stdlib.h>

#include "ff_type.hpp"

class Synchronize
{
public:
    struct Clock {
        int64_t start_time;
        int64_t current_pts;
    };

private:
    Clock audio;
    Clock video;
    Clock absolute;
    float ptsRatio;
    SynchronizeType type;

    //    int estimate_video_duration;
    //    int n;

public:
    explicit Synchronize(SynchronizeType _type);
    int64_t getCurrentTime();
    void reset();
    void setPtsRatio(float ratio);
    Clock& getMasterClock();
    int64_t getMasterTime();
    void setClockTime(SynchronizeType _type, int64_t pts);
    int64_t getClockTime(const Clock& clock);
    int updateVideo(int64_t pts, int64_t duration);
    int updateAudio(int samples, int samplerate, int64_t pts);
    int updateAudioByBytesSize(unsigned bytesSize, int samplerate, int channels,
                               int bitsPerSample, int64_t pts);
};


#endif
