# Copyright Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
from torch import multiprocessing as mp


class EplbPlannerProcess:
    def __init__(self, policy):
        self.policy = policy
        self.is_alive = True
        spawn_ctx = mp.get_context("spawn")
        self.plan_in = spawn_ctx.Queue()
        self.plan_out = spawn_ctx.Queue()
        self.plann_process = spawn_ctx.Process(target=EplbPlannerProcess._do_rebalance_experts_,
                                               args=(self.plan_in, self.plan_out))
        self.plann_process.start()

    @staticmethod
    def _do_rebalance_experts_(in_q, out_q):
        is_alive: bool = True

        def graceful_exit(signum, frame):
            nonlocal is_alive
            is_alive = False

        while True:
            if not is_alive:
                break
            args = in_q.get()
            if args == "exist":
                break
            policy, current_expert_table, expert_workload = args
            results = policy.rebalance_experts(current_expert_table, expert_workload)
            if is_alive:
                out_q.put((results))

    def shutdown(self):
        self.graceful_exit(0, 0)

    def graceful_exit(self, signum, frame):
        self.plan_in.put("exist")
        self.is_alive = False
        self.plann_process.terminate()

    def rebalance_experts(self, current_expert_table, expert_workload):
        if self.is_alive:
            self.plan_in.put((self.policy, current_expert_table, expert_workload))
            results = self.plan_out.get()
        else:
            raise Exception("planner process is shutdown")
        return results
