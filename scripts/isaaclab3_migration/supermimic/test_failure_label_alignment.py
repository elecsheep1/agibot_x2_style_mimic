"""核对回放失败标签与真实下一状态对齐，避免把终止标签提前一帧。"""
import importlib
import sys
import unittest
import torch
from test_failure_supervision import PREFIX

replay=importlib.import_module(PREFIX+'.algorithms.supermimic_tracking.replay_buffer')


class FailureLabelAlignmentTests(unittest.TestCase):
    def test_terminal_transition_is_labeled_once_at_its_target_state(self):
        buffer=replay.ReplayBuffer(1,16,75,29,35,'cpu')
        ids=torch.tensor([0])
        # s0、s1、s2 正常；a2 执行后抵达失败状态 s3。
        # record() 在转移行上存 next_done，另存一个终止状态行。
        for row in range(4):
            s=torch.zeros(1,75);s[:,3]=1;s[:,2]=.6 if row<3 else .2
            buffer.add(ids,s,torch.zeros(1,29),torch.zeros(1,29),torch.zeros(1,35),
                torch.zeros(1),torch.tensor([float(row>=2)]),torch.tensor([row]),ids,
                torch.tensor([row==3]),torch.tensor([row==0]))
        buffer.end_episode(ids)
        batch=buffer.sample_windows(1,3)
        torch.testing.assert_close(batch.states[:,0,2],torch.tensor([.6,.6,.6,.2]))
        actual_failure=(batch.states[1:,0,2]<.35).float()
        torch.testing.assert_close(batch.done[:,0],actual_failure,
            msg='失败标签必须属于 states[j+1]；不能提前到正常状态，也不能重复终止标记')


if __name__=='__main__':unittest.main()
