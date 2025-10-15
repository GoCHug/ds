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
#ifndef ATB_SPEED_MODELS_DEEPSEEK_V2_MTP_DECODER_MODEL_H
#define ATB_SPEED_MODELS_DEEPSEEK_V2_MTP_DECODER_MODEL_H
#include <atb/comm.h>
#include <vector>
#include "atb_speed/base/model.h"
#include "atb_speed/utils/model_factory.h"
#include "models/deepseekv2/layer/decoder_layer.h"
#include "models/moe/model/decoder_model.h"
#include "models/deepseekv2/model/decoder_model.h"

namespace atb_speed {
namespace deepseekV2 {

class MtpDecoderModel : public atb_speed::moe::MoeDecoderModel {
public:
    explicit MtpDecoderModel(const std::string &param);
    atb::Status InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                           std::vector<atb::TensorDesc> &outTensorDescs) override;

private:
    atb::Status AddWordEmbedding() override;
    atb::Status AddPositionalEmbedding() override;
    void SetLayerParam(DecoderLayerParam &layerParam, int64_t layerId);
    atb::Status AddLayerWeights(atb_speed::Model::Node &layerNode, size_t &inTensorId, const uint32_t layerId);
    atb::Status AddNodesBeforeLayer() override;
    atb::Status AddNodesAfterLayer() override;
    atb::Status AddSingleLayer(uint32_t layerId) override;
    atb::Status AddParallelHostWeight(atb_speed::Model::Node &layerNode, size_t &inTensorId);
    atb::Status AddPrefixCacheHostWeight(atb_speed::Model::Node &layerNode, size_t &inTensorId);
    atb::Status AddLayerHostWeight(atb_speed::Model::Node &layerNode, size_t &inTensorId, int layerId);
    atb::Status AddDenseTpHostWeight(atb_speed::Model::Node &layerNode, size_t &inTensorId, int layerId);
    atb::Status AddFinalNorm() override;
    atb::Status AddLmhead() override;
    uint32_t CalcWeightTensorSize() override;
    void ConstructInTensorMap() override;
    void ConstructInternalTensorMap() override;
    void ConstructOutTensorMap() override;
    atb::Status BindParamHostTensor(uint32_t nodeId) override;
    atb::TensorDesc GetLogitsDesc(const std::vector<atb::TensorDesc> &inTensorDescs, uint32_t logitsIndicesIdx);
    atb::Status AddENorm();
    atb::Status AddHNorm();
    atb::Status AddConcat();
    atb::Status AddLinear();
    int GetWeightCountPerLayer();
    bool enableExpertCumSumOutput = false;
    bool enableTopkOutput = false;
    atb::Status AddGatherFinalStateOut();
    atb::Status AddGatherAfterLmhead();
    atb::Status AddIndicesGatherAfterLmhead();
    atb::Status AddSliceFinalStateOut();
    DeepseekV2ModelParam param;
};

REGISTER_MODEL(deepseekV2, MtpDecoderModel);

}  // namespace deepseekV2
}  // namespace atb_speed
#endif