# Copyright Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
import queue


class EplbPlannerExecuter(object):
    def __init__(self):
        self.prepare_queue = queue.Queue()
        self.load_queue = queue.Queue()

    #for loader thread
    def set_load_prepare_done_and_wait(self):
        self.prepare_queue.put(1)
        self.load_queue.get()

    #for forward thread
    def is_load_prepare_done(self):
        return not self.prepare_queue.empty()

    #for forward thread
    def set_load_deploy_done_and_notify(self):
        self.load_queue.put_nowait(1)


