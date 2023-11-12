#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

#include "module/vi/module_fileReader.hpp"
#include "module/vp/module_inference.hpp"
#include "module/vp/module_mppdec.hpp"
#include "postprocess.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc.hpp"

struct External_ctx {
    shared_ptr<ModuleMedia> module;
    std::vector<rknn_tensor_attr*> output_attrs;
    std::vector<rknn_tensor_mem*> output_mems;
};

void callback_external(void* _ctx, shared_ptr<MediaBuffer> buffer)
{
    External_ctx* ctx = static_cast<External_ctx*>(_ctx);
    shared_ptr<VideoBuffer> buf = static_pointer_cast<VideoBuffer>(buffer);
    void* ptr = buf->getActiveData();
    //  printf("----------w = %u h = %u \n", buf->getImagePara().width, buf->getImagePara().height);
    uint32_t width = buf->getImagePara().hstride;
    uint32_t height = buf->getImagePara().vstride;
    float out_image_w = 1920.0;
    float out_image_h = 1080.0;
    float scale_w = (float)(out_image_w / width);
    float scale_h = (float)(out_image_h / height);
    cv::Mat imgRgb(cv::Size(width, height), CV_8UC3, ptr);
    cv::Mat imgBgr;
    cvtColor(imgRgb, imgBgr, cv::COLOR_RGB2BGR);
    cv::Mat resizeImg;
    cv::resize(imgBgr, resizeImg, cv::Size(out_image_w, out_image_h));
    const float nms_threshold = NMS_THRESH;
    const float box_conf_threshold = BOX_THRESH;
    detect_result_group_t detect_result_group;
    std::vector<float> out_scales;
    std::vector<int32_t> out_zps;
    for (uint32_t i = 0; i < ctx->output_attrs.size(); ++i) {
        out_scales.push_back(ctx->output_attrs[i]->scale);
        out_zps.push_back(ctx->output_attrs[i]->zp);
    }
    post_process((int8_t*)ctx->output_mems[0]->virt_addr, (int8_t*)ctx->output_mems[1]->virt_addr, (int8_t*)ctx->output_mems[2]->virt_addr,
                 height, width, box_conf_threshold, nms_threshold, scale_w, scale_h, out_zps, out_scales, &detect_result_group);

    char text[256];
    for (int i = 0; i < detect_result_group.count; i++) {
        detect_result_t* det_result = &(detect_result_group.results[i]);
        sprintf(text, "%s %.1f%%", det_result->name, det_result->prop * 100);
        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;
        rectangle(resizeImg, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0, 255), 3);
        putText(resizeImg, text, cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    cv::imshow(ctx->module->getName(), resizeImg);
    cv::waitKey(1);
}

// Usage： ./demo_rknn ./file.mp4 ./model/RK3588/yolov5s-640-640.rknn
int main(int argc, char** argv)
{
    int ret = -1;
    shared_ptr<ModuleFileReader> file_reader = NULL;
    shared_ptr<ModuleMppDec> dec = NULL;
    shared_ptr<ModuleInference> inf = NULL;

    // ImagePara output_para = {3840, 2160, 640, 640, V4L2_PIX_FMT_HEVC};
    ImagePara input_para;
    External_ctx* ctx1 = NULL;

    if (argc < 3) {
        ff_error("The number of parameters is incorrect\n");
        return ret;
    };

    do {
        file_reader = make_shared<ModuleFileReader>(argv[1], false);
        // file_reader->setOutputImagePara(output_para);
        ret = file_reader->init();
        if (ret < 0) {
            ff_error("file_reader init failed\n");
            break;
        }
        input_para = file_reader->getOutputImagePara();
       
        printf("----------w = %u h = %u \n", input_para.width, input_para.height);

        dec = make_shared<ModuleMppDec>(input_para);
        dec->setProductor(file_reader);
        ret = dec->init();
        if (ret < 0) {
            ff_error("Dec init failed\n");
            break;
        }

        input_para = dec->getOutputImagePara();
        printf("----------w = %u h = %u \n", input_para.width, input_para.height);
        inf = make_shared<ModuleInference>(input_para);
        inf->setProductor(dec);
        inf->setInferenceInterval(1);
        if (inf->setModelData(argv[2], 0) < 0) {
            ff_error("inf setModelData fail!\n");
            break;
        }
        ret = inf->init();
        if (ret < 0) {
            ff_error("inf init failed\n");
            break;
        }
        input_para = inf->getOutputImagePara();
        printf("----------w = %u h = %u hs = %u vs =  %u\n", input_para.width, input_para.height, input_para.hstride, input_para.vstride);
        ctx1 = new External_ctx();
        ctx1->module = inf;
        ctx1->output_attrs = inf->getOutputAttrRef();
        ctx1->output_mems = inf->getOutputMemRef();
        inf->setOutputDataCallback(ctx1, callback_external);

        file_reader->start();
        getchar();
        file_reader->stop();

    } while (0);

    if (ctx1)
        delete ctx1;
    return ret;
}
