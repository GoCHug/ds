/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef ATB_SPEED_MODELS_DEEPSEEK_V2_DECODER_MODEL_H
#define ATB_SPEED_MODELS_DEEPSEEK_V2_DECODER_MODEL_H
#include <atb/comm.h>
#include <vector>
#include "atb_speed/base/model.h"
#include "atb_speed/utils/model_factory.h"
#include "models/deepseekv2/layer/decoder_layer.h"
#include "models/moe/model/decoder_model.h"

namespace atb_speed {
namespace deepseekV2 {
class DeepseekV2ModelParam : public atb_speed::moe::MoeModelParam {
public:
    // MLA参数
    int qLoraRank = 1536;
    int kvLoraRank = 512;
    int headNum = 128;
    int qkNopeHeadDim = 128;
    int qkRopeHeadDim = 64;
    float softmaxScale = 0;
    int moePackQuantType = atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED;
    bool enableMlaPreprocess = false;
    bool isNzCache = false;
    bool enablePrefixCache = false;
    // Grouped topk参数
    int numOfGroups = 1;
    int scaledTopk = -1; /// 非deepseek模型默认不启用scaledTopk特性
    bool enableInitRoutingCutoff = false;  /// A flag indicating whether to use scaled topk option
    float routedScalingFactor = 1;
    std::string routingMethod = "deviceLimited";
    std::string processLogits = "scaling";
    atb::SVector<int32_t> topkGroups = {};
    bool enableFusedTopk = false;
    bool mlaOP = false;
    bool enableExtraOprojTp = false;
    bool enableATBGateMatmul = false;
    bool enableMlaPrefetch = false;
    bool enableDistributed = false;

    std::vector<std::vector<int>> attnLinearQuantType = {};
    std::vector<std::vector<int>> attnLinearTransposeType = {};
    std::vector<bool> kvcacheQuantLayers;

    bool hasP2DWeight = false;
    bool finalStateOut = false;
    bool enableAllToAllMC2 = true;
    bool enableGatherPreNorm = false;
    bool enableLoadBalance = false;
    bool enableEPWB = false;
    uint32_t numOfRedundantExpert = 0;
    int64_t numDanglingSharedExperts = 0;
    bool mixSharedRouting = false;
    bool enableExpertCumSumOutput = false;
    bool enableDenseTp = false;
    bool enableTopkOutput = false;

    bool maskfree = true;

    // h3p
    bool enableQkvdownDp = false;
    bool enableSharedExpertDp = false;
    bool enableGatingDp = false;
    bool enableSharedExpertOverlap = false;
    bool enableInfNan = true;
    bool enableLcocTp = false;
    bool enableLcocAll2All = false;
    bool enableFusedMLA = false;

    static HcclComm dispatchAndCombineHcclComm;
    static std::string dispatchAndCombinecommDomain;

    void FromString(const std::string &param);
    void AddParamJsonMLA(const std::string &param);
    void AddParamJsonMoE(const std::string &param);
    void AddParamJsonMoEGate(const std::string &param);
    void AddParamJsonH3P(const std::string &param);
    void CheckParam() override;
    void SetHcclComm() const;
    void AddLogInfo();
    void CheckMixParallelValid() const;
};

class DecoderModel : public atb_speed::moe::MoeDecoderModel {
public:
    explicit DecoderModel(const std::string &param);
    atb::Status InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                           std::vector<atb::TensorDesc> &outTensorDescs) override;

protected:
    DeepseekV2ModelParam param;
private:
    uint32_t CalcWeightTensorSize() override;
    atb::Status AddWordEmbedding() override;
    atb::Status AddPositionalEmbedding() override;
    void SetLayerParam(DecoderLayerParam &layerParam, int64_t layerId);
    atb::Status AddSequenceParallelHostWeight(atb_speed::Model::Node &layerNode, size_t &inTensorId);
    atb::Status AddPrefixCacheHostWeight(atb_speed::Model::Node &layerNode, size_t &inTensorId);
    atb::Status AddDenseTpHostWeight(atb_speed::Model::Node &layerNode, size_t &inTensorId, int layerId);
    atb::Status AddNodesBeforeLayer() override;
    atb::Status AddNodesAfterLayer() override;
    atb::Status AddSingleLayer(uint32_t layerId) override;
    atb::Status AddParallelHostWeight(atb_speed::Model::Node &layerNode, size_t &inTensorId);
    atb::Status AddLayerHostWeight(atb_speed::Model::Node &layerNode, size_t &inTensorId, int layerId);
    atb::Status AddFinalNorm() override;
    atb::Status AddLmhead() override;
    atb::Status AddGatherAfterLmhead();
    atb::Status AddGatherFinalStateOut();
    atb::Status AddSliceFinalStateOut();
    void ConstructInTensorMap() override;
    void ConstructInternalTensorMap() override;
    void ConstructOutTensorMap() override;
    atb::Status BindParamHostTensor(uint32_t nodeId) override;
    atb::TensorDesc GetLogitsDesc(const std::vector<atb::TensorDesc> &inTensorDescs, uint32_t logitsIndicesIdx);
};
REGISTER_MODEL(deepseekV2, DecoderModel);
void SetMlaParam(DecoderLayerParam &layerParam, const DeepseekV2ModelParam &param, int64_t layerId = 0);
void SetMoeParam(DecoderLayerParam &layerParam, const DeepseekV2ModelParam &param, int64_t layerId);
void SetParallelParam(DecoderLayerParam &layerParam, const DeepseekV2ModelParam &param);

}  // namespace deepseekV2
}  // namespace atb_speed
#endif
