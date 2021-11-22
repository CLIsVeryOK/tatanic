#ifndef _CW_PREDICT_TARGETS_H_
#define _CW_PREDICT_TARGETS_H_
#include <mutex>
#include "cw_predict_def.h"
#include "cw_predict_impl.h"
#include "config_util/read_config.h"

#if 0
/*
signature_def['common_detect']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['input_images'] tensor_info:
        dtype: DT_FLOAT
        shape: (16, 3, 384, 640)
        name: input_images:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['f1'] tensor_info:
        dtype: DT_FLOAT
        shape: (16, 40, 12, 20)
        name: Conv_24/BiasAdd:0
    outputs['f2'] tensor_info:
        dtype: DT_FLOAT
        shape: (16, 40, 24, 40)
        name: Conv_32/BiasAdd:0
    outputs['f3'] tensor_info:
        dtype: DT_FLOAT
        shape: (16, 40, 48, 80)
        name: Conv_40/BiasAdd:0
  Method name is: tensorflow/serving/predict
*/
#endif

#define CWDEF_COMMONDET_INPUTTENSORS        (2)

#define CWDEF_COMMONDET_MODELNAME           ("cw_common_detect_models")
#define CWDEF_COMMONDET_SIGNATURENAME       ("common_detect")
#define CWDEF_COMMONDET_MODELVERSION        (0)


#define CWDEF_COMMONDET_MAXBATCH            (16)

#define CWDEF_COMMONDET_INNAME              ("input_images")
// #define CWDEF_COMMONDET_INWIDTH             (640)
// #define CWDEF_COMMONDET_INHEIGHT            (384)
#define CWDEF_COMMONDET_INCHANNELS          (3)

// reuse output, ped+face+head model from chenjiapeng also use
// three output feature map, but channel is not the same
#define CWDEF_COMMONDET_F1ALIAS             ("f1")
// #define CWDEF_COMMONDET_F1WIDTH             (20)
// #define CWDEF_COMMONDET_F1HEIGHT            (12)
// #define CWDEF_COMMONDET_F1CHANNELS          (40)

#define CWDEF_COMMONDET_F2ALIAS             ("f2")
// #define CWDEF_COMMONDET_F2WIDTH             (40)
// #define CWDEF_COMMONDET_F2HEIGHT            (24)
// #define CWDEF_COMMONDET_F2CHANNELS          (40)

#define CWDEF_COMMONDET_F3ALIAS             ("f3")
// #define CWDEF_COMMONDET_F3WIDTH             (80)
// #define CWDEF_COMMONDET_F3HEIGHT            (48)
// #define CWDEF_COMMONDET_F3CHANNELS          (40)



namespace tensorflow {
namespace serving {

class cwPredictTargets         : public cwPredict{
 public:
  explicit cwPredictTargets(ServerCore* core)
    :cwPredict(core, CWDEF_COMMONDET_MODELNAME,CWDEF_COMMONDET_SIGNATURENAME,CWDEF_COMMONDET_MODELVERSION){
    _init();
    _vTensorNames.emplace_back(CWDEF_COMMONDET_F1ALIAS);
    _vTensorNames.emplace_back(CWDEF_COMMONDET_F2ALIAS);
    _vTensorNames.emplace_back(CWDEF_COMMONDET_F3ALIAS);
    //resizeAnchors();

    string config_file = "/serving/data/config/cw_config.cfg";
    std::map<string, string> mapConfig;
    ReadConfig(config_file, mapConfig);
    std::map<string, string>::iterator iter;
    for (iter = mapConfig.begin(); iter != mapConfig.end(); iter++) {
        if (!(iter->first).compare("detect_thresh")) {
            kfScoreThreld = atof(iter->second.c_str());
            VLOG(0) << "read detect thresh from config: " << kfScoreThreld;
        }
    }
  }

  ~cwPredictTargets(){
  }
    //==================================
    //Function：
    //  initTensors():初始化输入和输出Tensor
    //Parameter:
    //Return:
    //==================================
    Status warmup(const RunOptions& run_options,ServerCore* core);


    //==================================
    //Function：
    //  doPredict():执行推理
    //Parameter:
    // [IN] run_options: 执行配置
    // [IN] core: 执行推理所需要的关键数据
    // [IN] vCamareId: 通道号
    // [IN] vFrameId: 帧号
    // [IN|OUT] vTargets: 返回结果
    //Return:
    //==================================
    Status doPredict(const RunOptions& run_options,ServerCore* core,
                     const std::vector<std::pair<string, Tensor>> &inputTensors,
                     const std::vector<int32_t> &vCamareId,
                     const std::vector<int32_t> &vFrameId,
                     std::vector<cwTarget> &vTargets);


    //==================================
    //Function：
    //  doPredict():输入追踪数据
    //Parameter:
    // [IN] run_options: 执行配置
    // [IN] core: 执行推理所需要的关键数据
    // [IN] model_spec: 模型数据
    // [IN] request: 请求数据
    // [IN|OUT] response: 返回结果
    //Return:
    //==================================
    Status doPredict(const RunOptions& run_options,ServerCore* core,
                     const ModelSpec& model_spec,
                     const PredictRequest& request,
                     PredictResponse* response);

protected:


    //==================================
    //Function：
    //  resizeAnchors():计算Anchor大小，从416->640转换
    //Parameter:
    //Return:
    //==================================
    //Status resizeAnchors(void);

    //==================================
    //Function：
    //  initTensors():初始化输入和输出Tensor
    //Parameter:
    //Return:
    //==================================
    Status runPredict(const RunOptions& run_options,
                        const std::vector<int32_t> &vCamareId,
                        const std::vector<int32_t> &vFrameId,
                        const std::vector<std::pair<string, Tensor>> &inputs,
                        std::vector<cwTarget> &vAllTargets);

    //==================================
    //Function：
    //  postProcess():检测后处理，包括解析feature和nms
    //Parameter:
    // [IN] vCamareId: 通道号
    // [IN] vFrameId: 帧号
    // [IN] inputs: 检测输出Tensor
    // [IN|OUT] vAllTargets: 返回结果
    //Return:
    //==================================
    Status postProcess(const std::vector<int32_t> &vCamareId,
                        const std::vector<int32_t> &vFrameId,
                        const std::vector<std::pair<string, Tensor>> &inputs,
                        std::vector<cwTarget> &vAllTargets);

    //==================================
    //Function：
    //  parseLayoutPerImage():分层解析
    //Parameter:
    // [IN] uImgidx: 图像索引
    // [IN] uCamerId: 通道号
    // [IN] uFrameId: 帧号
    // [IN] inputs: 检测输出Tensor
    // [IN|OUT] vTargets: 返回结果
    //Return:
    //==================================
    Status parseLayoutPerImage(uint32_t uImgidx, uint32_t uCamerId, uint32_t uFrameId,
                               const std::vector<std::pair<string, Tensor>> &inputs,
                               std::vector<cwTarget> &vTargets);

    //==================================
    //Function：
    //  parseLayoutPerImageV2():分层解析重庆[行人/人头/人脸]检测模型
    //Parameter:
    // [IN] uImgidx: 图像索引
    // [IN] uCamerId: 通道号
    // [IN] uFrameId: 帧号
    // [IN] inputs: 检测输出Tensor
    // [IN|OUT] vTargets: 返回结果
    //Return:
    //==================================
    Status parseLayoutPerImageV2(uint32_t uImgidx, uint32_t uCamerId, uint32_t uFrameId,
                                 const std::vector<std::pair<string, Tensor>> &inputs,
                                 std::vector<cwTarget> &vTargets);

    //==================================
    //Function：
    //  getTargetResultTensors():填充返回结果
    //Parameter:
    // [IN] vTarget:            图像索引
    // [IN|OUT] tensorCord:     框坐标
    // [IN|OUT] tensorProb:     框得分
    // [IN|OUT] tensorClass:    框类别
    // [IN|OUT] tensorBatchIdx: 框batch_idx
    //Return:
    //==================================
    Status getTargetResultTensors(const std::vector<cwTarget> &vTarget,
                                  Tensor &tensorCord,
                                  Tensor &tensorProb,
                                  Tensor &tensorClass,
                                  Tensor &tensorBatchIdx);


protected:
    //后处理接口
    float sigmod(float fx);
    float sigmod_inv(float fthreld);
    float clamp(const float val, const float minVal, const float maxVal);
    std::vector<cwTarget> nonMaximumSuppression(const float nmsThresh, std::vector<cwTarget> binfo);
    std::vector<cwTarget> nmsAllClasses(const float nmsThresh, std::vector<cwTarget>& binfo,const unsigned numClasses);

    std::vector<std::string> _vTensorNames;

private:
    //float kfScoreThreld = 0.005f;
    float kfScoreThreld = 0.6f;
    float kfNumThreld = 0.45f;
    const uint32_t kAnchors = 4;
    const uint32_t kClasses = 5;
    const uint32_t kClassesV2 = 3; // model from chenjiapeng, /ped/face/head/
    const uint32_t kAnchorsV2 = 3; // model from chenjiapeng, /ped/face/head/
    const uint32_t kWidth = CWDEF_COMMONDET_INWIDTH;
    const uint32_t kHeight = CWDEF_COMMONDET_INHEIGHT;

    //std::vector<std::pair<string, Tensor>> _inputTensor;
    //std::vector<std::pair<string, Tensor>> _outputTensor;

    std::mutex _runPredictMtx;
};

}  // namespace serving
}  // name

#endif //_CW_PREDICT_FRAMES_H_
