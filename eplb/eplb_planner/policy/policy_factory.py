# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from .eplb_policy import EplbPolicy, DynamicConfig
from .mock_load_balance import MockLoadBalance
from .dynamic_ep import DynamicEP
from .flash_lb import FlashLB


class PolicyFactory:
    @staticmethod
    def generate_policy(policy_type: int, config: DynamicConfig) -> EplbPolicy:
        policy = {
            0: MockLoadBalance,
            1: DynamicEP,
            2: FlashLB,
        }
        return policy.get(policy_type, MockLoadBalance)(config)
