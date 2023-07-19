#ifndef __MODULE_RTSPCLIENT_HPP__
#define __MODULE_RTSPCLIENT_HPP__

#include "module/module_media.hpp"
class RTSPClient;

class ModuleRtspClient : public ModuleMedia
{
public:
    ModuleRtspClient(const char* url, RTSP_STREAM_TYPE _stream_type = RTSP_STREAM_TYPE_UDP,
                     bool enable_video = true, bool enable_audio = false);
    ~ModuleRtspClient();
    int init();
    const uint8_t* videoExtraData();
    unsigned videoExtraDataSize();
    const uint8_t* audioExtraData();
    unsigned audioExtraDataSize();
    int audioChannel();
    int audioSampleRate();
    uint32_t videoFPS();
    void setTimeOutSec(unsigned sec, unsigned nsec) { time_msec = sec * 1000 + nsec / 1000; }
    void setMaxTimeOutCount(int count) { maxTimeOutCount = count; }

protected:
    virtual ProduceResult doProduce(shared_ptr<MediaBuffer> output_buffer) override;
    virtual void bufferReleaseCallBack(shared_ptr<MediaBuffer> buffer) override;
    virtual bool setup() override;
    virtual bool teardown() override;
    int fromRtpGetVideoParameter();
    static void closeHandlerFunc(void* arg, int err, int result);

private:
    shared_ptr<RTSPClient> rtsp_client;
    int64_t time_msec;
    RTSP_STREAM_TYPE stream_type;
    string url;
    int abnormalStatusFlag;
    int timeOutCount;
    int maxTimeOutCount;

    bool open();
};

#endif /* ModuleRtspClient_hpp */
