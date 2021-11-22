#include "cw_predict_targets.h"

#include <string>
#include <utility>
#include <chrono>

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow_serving/core/servable_handle.h"
#include "tensorflow_serving/servables/tensorflow/predict_util.h"
#include "tensorflow_serving/servables/tensorflow/util.h"

namespace tensorflow {
namespace serving {


static const float _strides[]    = { 32, 16, 8 };
/*
static const float _anchors[][2] = {
    { 144, 172 },    { 171, 312 },    { 301, 184 },     { 351, 354 },
    { 36, 110 },     { 84, 118 },     { 180, 77 },      { 72, 238 },
    { 10, 16 },      { 20, 48 },      { 44, 25 },       { 69, 57 }
};
*/

static const float _anchors[][2] = {
    { 221.538467, 264.615387 },    { 263.076935, 480.000000 },    { 463.076935, 283.076935 },     { 540.000000, 544.615417 },
    { 55.3846169, 169.230774 },    { 129.230774, 181.538467 },    { 276.923096, 118.461540 },     { 110.769234, 366.153839 },
    { 15.3846159, 24.6153851 },    { 30.7692318, 73.8461533 },    { 67.6923065, 38.4615402 },     { 106.153847, 87.6923065 }
};

// /ped/face/head/ model anchor, 3*3 for each class
static const float _anchors_V2[][2] = {
    {77.739, 98.754}, {104.533, 194.179}, {300.445, 439.172},  // ped, 12*20
    {39.164, 42.026}, {26.589,  70.962} , {43.103,  103.929},  // ped, 24*40
    {7.036,  12.776}, {12.637,  25.363},  {17.654,  44.718},   // ped, 48*80
    {40.504, 53.595}, {59.008,  77.596},  {93.845,  119.045},  // face, 12*20
    {12.111, 16.357}, {17.430,  20.512},  {21.719,  27.493},   // face, 24*40
    {4.443,  5.309},  {6.0569,  8.059},   {8.756,   11.332},   // face, 48*80
    {36.364, 43.846}, {66.583,  80.830},  {113.175, 137.420},  // head, 12*20
    {11.800, 13.563}, {17.121,  19.865},  {24.732,  29.662},   // head, 24*40
    {4.280,  4.679},  {6.100,   6.802},   {8.401,   9.535},    // head, 48*80
};

// detect model from yufufu <--> detect model from chenjiapeng
// [class 0/ped]            <--> [class 0/ped]
// [class 1/non_motor]      <--> [-]
// [class 2/motor]          <--> [-]
// [class 3/face]           <--> [class 1/face]
// [class 4/tricycle]       <--> [-]
// [-]                      <--> [class 2/head]
static const int class_ref_table[] = {0, 3, 5};

float cwPredictTargets::sigmod(float fx)
{
    return 1.0f / (1.0f + exp(-fx));
}

float cwPredictTargets::sigmod_inv(float fthreld)
{
    return -log(1.0f / fthreld - 1.0f);
}

float cwPredictTargets::clamp(const float val, const float minVal, const float maxVal)
{
    assert(minVal <= maxVal);
    return std::min(maxVal, std::max(minVal, val));
}


std::vector<cwTarget> cwPredictTargets::nonMaximumSuppression(const float nmsThresh, std::vector<cwTarget> binfo)
{
    auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float {
        if (x1min > x2min)
        {
            std::swap(x1min, x2min);
            std::swap(x1max, x2max);
        }
        return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
    };
    auto computeIoU = [&overlap1D](cwTarget& t1, cwTarget& t2) -> float {
        float overlapX = overlap1D(t1.fX1, t1.fX2, t2.fX1, t2.fX2);
        float overlapY = overlap1D(t1.fY1, t1.fY2, t2.fY1, t2.fY2);
        float area1 = (t1.fX2 - t1.fX1) * (t1.fY2 - t1.fY1);
        float area2 = (t2.fX2 - t2.fX1) * (t2.fY2 - t2.fY1);
        float overlap2D = overlapX * overlapY;
        float u = area1 + area2 - overlap2D;
        return u == 0 ? 0 : overlap2D / u;
    };

    std::stable_sort(binfo.begin(), binfo.end(),
        [](const cwTarget& t1, const cwTarget& t2) { return t1.fProb > t2.fProb; });
    std::vector<cwTarget> out;
    for (auto& i : binfo)
    {
        bool keep = true;
        for (auto& j : out)
        {
            if (keep)
            {
                float overlap = computeIoU(i, j);
                keep = overlap <= nmsThresh;
            }
            else
                break;
        }
        if (keep) out.push_back(i);
    }
    return out;
}

std::vector<cwTarget> cwPredictTargets::nmsAllClasses(const float nmsThresh, std::vector<cwTarget>& binfo,
    const unsigned numClasses)
{
    std::vector<cwTarget> result;
    std::vector<std::vector<cwTarget>> splitBoxes(numClasses);
    for (auto& box : binfo)
    {
        splitBoxes.at(box.uClsId).push_back(box);
    }

    for (auto& boxes : splitBoxes)
    {
        boxes = nonMaximumSuppression(nmsThresh, boxes);
        result.insert(result.end(), boxes.begin(), boxes.end());
    }

    return result;
}
/*
//=================================================================================================//
//==================================
//Function：
//  resizeAnchors():计算Anchor大小，从416->640转换
//Parameter:
//Return:
//==================================
Status cwPredictTargets::resizeAnchors(void)
{
    float fRatio = 640.0f/416.0f;
    uint32_t uAnchors = 3 * kAnchors;
    for(uint32_t i = 0; i < uAnchors; ++i){
        _anchors[i][0] *= fRatio;
        _anchors[i][1] *= fRatio;
    }
    return Status::OK();
}


//==================================
//Function：
//  initTensors():初始化输入和输出Tensor
//Parameter:
//Return:
//==================================
Status cwPredictTargets::initTensors(void)
{
    //output tensor
    TensorShape f1Shape({CWDEF_COMMONDET_MAXBATCH, CWDEF_COMMONDET_F1CHANNELS, CWDEF_COMMONDET_F1HEIGHT,CWDEF_COMMONDET_F1WIDTH});
    TensorShape f2Shape({CWDEF_COMMONDET_MAXBATCH, CWDEF_COMMONDET_F2CHANNELS, CWDEF_COMMONDET_F2HEIGHT,CWDEF_COMMONDET_F2WIDTH});
    TensorShape f3Shape({CWDEF_COMMONDET_MAXBATCH, CWDEF_COMMONDET_F3CHANNELS, CWDEF_COMMONDET_F3HEIGHT,CWDEF_COMMONDET_F3WIDTH});

    Tensor f1Tensor(DT_FLOAT, f1Shape);
    Tensor f2Tensor(DT_FLOAT, f2Shape);
    Tensor f3Tensor(DT_FLOAT, f3Shape);

    _outputTensor.emplace_back(std::make_pair(CWDEF_COMMONDET_F1ALIAS, f1Tensor));
    _outputTensor.emplace_back(std::make_pair(CWDEF_COMMONDET_F2ALIAS, f2Tensor));
    _outputTensor.emplace_back(std::make_pair(CWDEF_COMMONDET_F3ALIAS, f3Tensor));


    //input tensor
    TensorShape inShape({CWDEF_COMMONDET_MAXBATCH,CWDEF_COMMONDET_INHEIGHT,CWDEF_COMMONDET_INWIDTH,CWDEF_COMMONDET_INCHANNELS,});
    Tensor inTensor(DT_UINT8, inShape);
    _inputTensor.emplace_back(std::make_pair(CWDEF_COMMONDET_INNAME, inTensor));

    return Status::OK();
}
*/
//==================================
//Function：
//  initTensors():初始化输入和输出Tensor
//Parameter:
//Return:
//==================================
Status cwPredictTargets::warmup(const RunOptions& run_options,ServerCore* core)
{
    Status status;
    std::vector<std::pair<string, Tensor>> inputs;
    std::vector<std::pair<string, Tensor>> outputs;
    Tensor inTensor(DT_UINT8, TensorShape({CWDEF_COMMONDET_MAXBATCH,CWDEF_COMMONDET_INHEIGHT,CWDEF_COMMONDET_INWIDTH,CWDEF_COMMONDET_INCHANNELS}));
    inputs.emplace_back(std::make_pair(CWDEF_COMMONDET_INNAME, inTensor));

    status = _runPredict(run_options,inputs,outputs);
    if (!status.ok()) {
        VLOG(0) << status.ToString();
        return status;
    }
    VLOG(0)<<"cwPredictTargets::warmup ok";
    return status;

}

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
Status cwPredictTargets::postProcess(const std::vector<int32_t> &vCamareId,
                                    const std::vector<int32_t> &vFrameId,
                                    const std::vector<std::pair<string, Tensor>> &inputs,
                                    std::vector<cwTarget> &vAllTargets)
{
    uint32_t uImgs = vCamareId.size();
    for(uint32_t img = 0; img < uImgs; ++img){
        std::vector<cwTarget> vTargets;

        auto t_start = std::chrono::high_resolution_clock::now();

        parseLayoutPerImageV2(img, vCamareId[img], vFrameId[img], inputs, vTargets);

        auto t_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start);
        VLOG(0) << "nms cost: " << duration.count() << "us";
        VLOG(0) << "finish process img: " << img;

        if(!vTargets.empty()){
            vAllTargets.insert(vAllTargets.end(), vTargets.begin(),vTargets.end());
        }
    }
    return Status::OK();
}

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
Status cwPredictTargets::parseLayoutPerImage(uint32_t uImgidx, uint32_t uCamerId, uint32_t uFrameId,
                                        const std::vector<std::pair<string, Tensor>> &inputs,
                                        std::vector<cwTarget> &vTargets)
{
    std::vector<cwTarget> vAllTargets;
    float fInvScoreThreld = sigmod_inv(kfScoreThreld);

    for(uint32_t idx = 0; idx < inputs.size(); ++idx){
        const Tensor &tensor = inputs[idx].second;

        //uint32_t n = tensor.dim_size(0);
        uint32_t c = tensor.dim_size(1);
        uint32_t h = tensor.dim_size(2);
        uint32_t w = tensor.dim_size(3);

        //VLOG(0)<<"uImgidx:"<<uImgidx<<" n:"<<n<<" c:"<<c<<" h:"<<h<<" w:"<<w;

        const float *fFeatures = tensor.flat<float>().data();
        //VLOG(0)<<"parseLayoutPerImage fFeatures "<<fFeatures;
        //const uint32_t uFeatureLen = tensor.NumElements();


        for (uint32_t a = 0; a < kAnchors; ++a) {
            uint32_t pos = uImgidx * c * h * w + a * 10 * h * w;
            for (uint32_t i = 0; i < h; ++i) {
                for (uint32_t j = 0; j < w; ++j) {
                    float fProb = 1.0f;
                    if(fFeatures[pos + 4 * h * w + i*w + j] < fInvScoreThreld){
                        continue;
                    }
                    fProb = sigmod(fFeatures[pos + 4 * h * w + i*w + j]);//objness
                    if (fProb < kfScoreThreld) {
                        continue;
                    }

                    float fSumCls = 0.0f;
                    std::vector<float> vCls;
                    vCls.resize(kClasses);
                    for (uint32_t cls = 0; cls < kClasses; ++cls) {
                        vCls[cls] = exp(fFeatures[pos + (5 + cls) * h * w + i*w + j]);
                        fSumCls += vCls[cls];
                    }

                    std::vector<cwTarget> vCurTargets;
                    vCurTargets.clear();
                    for(uint32_t cls = 0; cls < kClasses; ++cls){
                        float fSoftMaxCls = vCls[cls]/fSumCls;
                        if (fProb * fSoftMaxCls < kfScoreThreld) {
                            continue;
                        }
                        cwTarget inferTarget;
                        memset(&inferTarget, 0, sizeof(cwTarget));
                        inferTarget.fProb = fProb * fSoftMaxCls;
                        inferTarget.uClsId = cls;
                        vCurTargets.emplace_back(inferTarget);
                    }

                    //VLOG(0)<<"fFeatures["<<pos + 0 * h * w + i*w + j<<"]="<<fFeatures[pos + 0 * h * w + i*w + j];
                    if(!vCurTargets.empty()){
                        float fCentor_x = (sigmod(fFeatures[pos + 0 * h * w + i*w + j]) + j) * _strides[idx];//x
                        float fCentor_y = (sigmod(fFeatures[pos + 1 * h * w + i*w + j]) + i) * _strides[idx];//y
                        float fScale_x = (exp(fFeatures[pos + 2 * h * w + i*w + j]))* _anchors[idx * kAnchors + a][0];//w
                        float fScale_y = (exp(fFeatures[pos + 3 * h * w + i*w + j]))* _anchors[idx * kAnchors + a][1];//h
                        float fX1 = clamp((fCentor_x - fScale_x / 2.0f)/ kWidth,0.0f, 1.0f);
                        float fX2 = clamp((fCentor_x + fScale_x / 2.0f)/ kWidth,0.0f, 1.0f);
                        float fY1 = clamp((fCentor_y - fScale_y / 2.0f)/ kHeight,0.0f, 1.0f);
                        float fY2 = clamp((fCentor_y + fScale_y / 2.0f)/ kHeight,0.0f, 1.0f);

                        for(uint32_t n = 0; n < vCurTargets.size(); ++n){
                            vCurTargets[n].fX1 = fX1;
                            vCurTargets[n].fX2 = fX2;
                            vCurTargets[n].fY1 = fY1;
                            vCurTargets[n].fY2 = fY2;
                        }
                        vAllTargets.insert(vAllTargets.end(), vCurTargets.begin(),vCurTargets.end());
                    }

                }
            }
        }
    }

    vTargets = nmsAllClasses(kfNumThreld, vAllTargets, kClasses);
    //VLOG(0)<<"imgidx"<<uImgidx<<" target:"<<vTargets.size();
    for(uint32_t i = 0; i < vTargets.size(); ++i){
        cwTarget &inferTarget = vTargets[i];
        // inferTarget.uClsId += 1;
        inferTarget.uBatchIdx = uImgidx;
        inferTarget.uCamerId = uCamerId;
        inferTarget.uFrameId = uFrameId;
    }
    return Status::OK();
}


Status cwPredictTargets::parseLayoutPerImageV2(uint32_t uImgidx, uint32_t uCamerId, uint32_t uFrameId,
                                               const std::vector<std::pair<string, Tensor>> &inputs,
                                               std::vector<cwTarget> &vTargets)
{
    std::vector<cwTarget> vAllTargets;
    float fInvScoreThreld = sigmod_inv(kfScoreThreld);
    for(uint32_t idx = 0; idx < inputs.size(); ++idx){
        const Tensor &tensor = inputs[idx].second;
        uint32_t c = tensor.dim_size(1); // =45
        uint32_t h = tensor.dim_size(2);
        uint32_t w = tensor.dim_size(3);

        // VLOG(0)<<"uImgidx:" << uImgidx << " c:" << c << " h:" << h << " w:"<< w;

        const float *fFeatures = tensor.flat<float>().data();
        const uint32_t idx_offset = uImgidx * c * h * w;
        for (uint32_t cls_idx = 0; cls_idx < kClassesV2; ++cls_idx) {
            for (uint32_t a = 0; a < kAnchorsV2; ++a) {
                uint32_t pos = idx_offset + (cls_idx*12 + a*4)*h*w;       // 检测框坐标相对feature map 的偏移量
                uint32_t pos_score = idx_offset + (36 + cls_idx*3 + a)*h*w; // 检测框得分相对feature map 的偏移量
                for (uint32_t i = 0; i < h; ++i) {
                    for (uint32_t j = 0; j < w; ++j) {
                        float fProb = 1.0f;
                        if(fFeatures[pos_score + i*w + j] < fInvScoreThreld){
                            continue;
                        }
                        fProb = sigmod(fFeatures[pos_score + i*w + j]);//objness

                        std::vector<cwTarget> vCurTargets;
                        vCurTargets.clear();
                        cwTarget inferTarget;
                        memset(&inferTarget, 0, sizeof(cwTarget));
                        inferTarget.fProb = fProb;
                        inferTarget.uClsId = cls_idx;
                        vCurTargets.emplace_back(inferTarget);

                        if(!vCurTargets.empty()){
                            float fCentor_x = (sigmod(fFeatures[pos + 0 * h * w + i*w + j]) + j) * _strides[idx];//x
                            float fCentor_y = (sigmod(fFeatures[pos + 1 * h * w + i*w + j]) + i) * _strides[idx];//y
                            float fScale_x = (exp(fFeatures[pos + 2 * h * w + i*w + j]))* _anchors_V2[cls_idx * 9 + idx * kAnchorsV2 + a][0];//w
                            float fScale_y = (exp(fFeatures[pos + 3 * h * w + i*w + j]))* _anchors_V2[cls_idx * 9 + idx * kAnchorsV2 + a][1];//h
                            float fX1 = clamp((fCentor_x - fScale_x / 2.0f)/ kWidth, 0.0f, 1.0f);
                            float fX2 = clamp((fCentor_x + fScale_x / 2.0f)/ kWidth, 0.0f, 1.0f);
                            float fY1 = clamp((fCentor_y - fScale_y / 2.0f)/ kHeight,0.0f, 1.0f);
                            float fY2 = clamp((fCentor_y + fScale_y / 2.0f)/ kHeight,0.0f, 1.0f);

                            // DEBUG CODE
                            // if (fY1 == 0.) {
                            //     VLOG(0) << (fCentor_x - fScale_x / 2.0f) << ", " << (fCentor_y - fScale_y / 2.0f);
                            //     VLOG(0) << idx << ", " << cls_idx << ", " << a << ", " << i << ", " << j;
                            //     VLOG(0) << fFeatures[pos + 0 * h * w + i*w + j] << ", " << fFeatures[pos + 1 * h * w + i*w + j] << ", " << fFeatures[pos + 2 * h * w + i*w + j] << ", " << fFeatures[pos + 3 * h * w + i*w + j];
                            //     VLOG(0) << _strides[idx] << ", " << _anchors_V2[cls_idx * 9 + idx * kAnchorsV2 + a][0] << ", " << _anchors_V2[cls_idx * 9 + idx * kAnchorsV2 + a][1];
                            //     VLOG(0) << vCurTargets[0].fProb << "," << vCurTargets[0].uClsId;
                            //     VLOG(0) << fX1 << ", " << fX2 << ", " << fY1 << ", " << fY2;
                            // }

                            for(uint32_t n = 0; n < vCurTargets.size(); ++n){
                                vCurTargets[n].fX1 = fX1;
                                vCurTargets[n].fX2 = fX2;
                                vCurTargets[n].fY1 = fY1;
                                vCurTargets[n].fY2 = fY2;
                            }
                            vAllTargets.insert(vAllTargets.end(), vCurTargets.begin(),vCurTargets.end());
                        }
                    }
                }
            }
        }
    }

    VLOG(0) << "imgidx" << uImgidx << " target.size(): " << vAllTargets.size();
    vTargets = nmsAllClasses(kfNumThreld, vAllTargets, kClassesV2);
     VLOG(0) << " vTargets.size():" << vTargets.size();
    for(uint32_t i = 0; i < vTargets.size(); ++i){
        cwTarget &inferTarget = vTargets[i];
        inferTarget.uBatchIdx = uImgidx;
        inferTarget.uCamerId = uCamerId;
        inferTarget.uFrameId = uFrameId;
        inferTarget.uClsId = class_ref_table[inferTarget.uClsId]; // match model v1.0
    }
    return Status::OK();
}

//==================================
//Function：
//  initTensors():初始化输入和输出Tensor
//Parameter:
//Return:
//==================================
Status cwPredictTargets::runPredict(const RunOptions& run_options,
                                const std::vector<int32_t> &vCamareId,
                                const std::vector<int32_t> &vFrameId,
                                const std::vector<std::pair<string, Tensor>> &inputs,
                                std::vector<cwTarget> &vAllTargets)
{
    Status status;
    std::vector<std::pair<string, Tensor>> outputs;
    std::vector<std::pair<string, Tensor>> tensorParse;

    std::unique_lock<std::mutex> lck(_runPredictMtx);

    status = _runPredict(run_options, inputs, outputs);
    if(!status.ok()){
       VLOG(0)<<"cwPredictTargets::runPredict failed";
       return status;
    }

    for(uint32_t i= 0; i < _vTensorNames.size(); ++i){
        for(uint32_t j = 0; j < outputs.size(); ++j){
            if(outputs[j].first == _vTensorNames[i]){
                tensorParse.emplace_back(std::make_pair(_vTensorNames[i], outputs[j].second));
                break;
            }
        }
    }

    return postProcess(vCamareId, vFrameId,tensorParse,vAllTargets);
}

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
Status cwPredictTargets::doPredict(const RunOptions& run_options,ServerCore* core,
                                    const std::vector<std::pair<string, Tensor>> &inputTensors,
                                    const std::vector<int32_t> &vCamareId,
                                    const std::vector<int32_t> &vFrameId,
                                    std::vector<cwTarget> &vAllTargets)
{
    // discard code
    // Status status;
    // uint32_t uBatchSize = 0;
    // std::vector<std::pair<string, Tensor>> inputs;

    //1.解析输入Tensorproto
    // for(uint32_t i = 0; i < inputTensors.size(); ++i){
    //      const string &alias = inputTensors[i].first;
    //      const TensorProto &proto = inputTensors[i].second;

    //      if(alias == CWDEF_COMMONDET_INNAME ){
    //         if(proto.dtype() != DT_UINT8){
    //             return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
    //                                       "input_images type should be uint8");
    //         }
    //         TensorShape shape(proto.tensor_shape());
    //         uBatchSize = shape.dim_size(0);

    //         Tensor tensorInput;
    //         if (!tensorInput.FromProto(proto)) {
    //             return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
    //                                 "tensor parsing error: " + alias);
    //         }
    //         inputs.emplace_back(std::make_pair(alias, tensorInput));
    //         break;
    //       }
    //  }

    // VLOG(0)<<"cwPredictTargets::doPredict batchsize:"<<uBatchSize;

    // //2.生成原图数组
    // {
    //     uint8_t* pInputImgs =  inputs[0].second.flat<uint8_t>().data();
    //     if(!pInputImgs){
    //         VLOG(0)<<"GetInputBatchImages pInputImgs nullptr ";
    //         return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
    //                                    "No camareids or frameids");
    //     }
    //     uint32_t uImgSize = CWDEF_COMMONDET_INCHANNELS*CWDEF_COMMONDET_INWIDTH*CWDEF_COMMONDET_INHEIGHT;
    //     for(uint32_t i = 0; i < vCamareId.size(); ++i){
    //          uint8_t* pcurImg = pInputImgs + i*uImgSize;

    //          cv::Mat cvImg(cv::Size(CWDEF_COMMONDET_INWIDTH,CWDEF_COMMONDET_INHEIGHT),CV_8UC3);
    //          memcpy(cvImg.data, pcurImg, uImgSize);

    //          // save image to file
    //          // cv::Mat bgr;
    //          // cv::cvtColor(cvImg, bgr, cv::COLOR_RGB2BGR);
    //          // cv::imwrite(strings::StrCat("image_src_", std::to_string(i), ".jpg"), bgr);

    //          vImages.emplace_back(cvImg);
    //     }
    // }

    //3.执行推理
    return runPredict(run_options, vCamareId, vFrameId, inputTensors, vAllTargets);

}


//==================================
//Function：
//  getTargetResultTensors():构造目标检测返回tensor
//Parameter:
// [IN] vTarget: 待构造目标集合
// [IN|OUT] tensorCord: 坐标
// [IN|OUT] tensorProb:置信度
// [IN|OUT] tensorBatchIdx:batch索引
//Return:
//==================================
Status cwPredictTargets::getTargetResultTensors(const std::vector<cwTarget> &vTarget,
                                                Tensor &tensorCord,
                                                Tensor &tensorProb,
                                                Tensor &tensorClass,
                                                Tensor &tensorBatchIdx)
{
    if(vTarget.empty()){
        return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                                  "No Targets specify");
    }
    //坐标
    uint32_t uTargetCount = (uint32_t)vTarget.size();
    tensorCord = Tensor(DT_FLOAT, TensorShape({uTargetCount,4}));

    //预测置信度
    tensorProb = Tensor(DT_FLOAT, TensorShape({uTargetCount,1}));

    //类别
    tensorClass = Tensor(DT_INT32, TensorShape({uTargetCount,1}));

    //Batch idx
    tensorBatchIdx = Tensor(DT_FLOAT, TensorShape({uTargetCount,1}));

    for(uint32_t i = 0; i < uTargetCount; ++i){
        const cwTarget &target = vTarget[i];
        float* CoordBuffer = tensorCord.flat<float>().data();
        float* ScoreBuffer = tensorProb.flat<float>().data();
        float* batchIdxBuffer = tensorBatchIdx.flat<float>().data();

        int32_t* clsBuffer = tensorClass.flat<int32_t>().data();

        CoordBuffer[i*4 + 0] = target.fY1;
        CoordBuffer[i*4 + 1] = target.fX1;
        CoordBuffer[i*4 + 2] = target.fY2;
        CoordBuffer[i*4 + 3] = target.fX2;

        ScoreBuffer[i] = target.fProb;
        batchIdxBuffer[i] = target.uBatchIdx;

        clsBuffer[i] = target.uClsId;
    }
    return Status::OK();

}

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
Status cwPredictTargets::doPredict(const RunOptions& run_options,ServerCore* core,
                                   const ModelSpec& model_spec,
                                   const PredictRequest& request,
                                   PredictResponse* response)
{
    Status status;
    uint32_t uBatchSize = 0;
    std::vector<std::pair<string, Tensor>> inputs;
    //1.解析输入Tensorproto
    for (auto& input : request.inputs()) {
      const string& alias = input.first;
      const TensorProto &proto = input.second;

      if(alias == CWDEF_COMMONDET_INNAME ){
         if(proto.dtype() != DT_UINT8){
             return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                                       "input_images type should be uint8");
         }
         TensorShape shape(proto.tensor_shape());
         uBatchSize = shape.dim_size(0);

         Tensor tensorInput;
         if (!tensorInput.FromProto(proto)) {
             return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                                 "tensor parsing error: " + alias);
         }
         inputs.emplace_back(std::make_pair(alias, tensorInput));
         break;
       }

    }

    VLOG(0)<<"cwPredictTargets::doPredict uBatchSize:"<<uBatchSize;
    if(uBatchSize < 1){
        return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                            "uBatchSize < 1 ");
    }

    //2.执行推理
    std::vector<int32_t> vCamareId;
    std::vector<int32_t> vFrameId;
    std::vector<cwTarget> vAllTargets;

    vCamareId.resize(uBatchSize);
    vFrameId.resize(uBatchSize);

    status = runPredict(run_options, vCamareId,vFrameId,inputs,vAllTargets);
    if (!status.ok()) {
        VLOG(0)<<"cwPredictTargets::doPredict failed";
        return status;
    }

    if(vAllTargets.empty()){
        Tensor tensorCord(DT_FLOAT, TensorShape({0}));
        Tensor tensorProb(DT_FLOAT, TensorShape({0}));
        Tensor tensorClass(DT_INT32, TensorShape({0}));
        Tensor tensorBatchIdx(DT_FLOAT, TensorShape({0}));

        tensorCord.AsProtoField(&((*response->mutable_outputs())["output_boxes"]));
        tensorProb.AsProtoField(&((*response->mutable_outputs())["output_scores"]));
        tensorClass.AsProtoField(&((*response->mutable_outputs())["output_classes"]));
        tensorBatchIdx.AsProtoField(&((*response->mutable_outputs())["output_batch_id"]));

    }else{
        Tensor tensorCord;
        Tensor tensorProb;
        Tensor tensorClass;
        Tensor tensorBatchIdx;

        getTargetResultTensors(vAllTargets, tensorCord, tensorProb, tensorClass,tensorBatchIdx);

        tensorCord.AsProtoField(&((*response->mutable_outputs())["output_boxes"]));
        tensorProb.AsProtoField(&((*response->mutable_outputs())["output_scores"]));
        tensorClass.AsProtoField(&((*response->mutable_outputs())["output_classes"]));
        tensorBatchIdx.AsProtoField(&((*response->mutable_outputs())["output_batch_id"]));
    }

    return Status::OK();


}

}
}

