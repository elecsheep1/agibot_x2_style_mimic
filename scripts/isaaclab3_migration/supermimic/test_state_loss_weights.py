"""分块损失的旧行为兼容、梯度与日志检查。"""
import unittest
from types import SimpleNamespace
import torch
from test_failure_supervision import wm, heads, state, reference


class StateLossWeightTests(unittest.TestCase):
    def model(self, weights=(1.,)*6):
        return wm.WorldModel(wm.WorldModelCfg(hidden=(16,),state_loss_weights=weights),
            SimpleNamespace(termination_features=heads.AnalyticHeads.termination_features),'cpu')

    def test_default_matches_original_value_and_gradient(self):
        model=self.model()
        error=torch.randn(2,3,74,requires_grad=True)
        loss=model.state_prediction_loss(error.square()).mean()
        original=error.square().sum(-1).mean()
        torch.testing.assert_close(loss,original)
        torch.testing.assert_close(torch.autograd.grad(loss,error,retain_graph=True)[0],
                                   torch.autograd.grad(original,error)[0])

    def test_weights_only_scale_selected_blocks(self):
        model=self.model((1.,1.,2.,2.,1.,1.))
        squared=torch.ones(2,74,requires_grad=True)
        torch.testing.assert_close(model.state_prediction_loss(squared),torch.full((2,),80.))
        model.state_prediction_loss(squared).sum().backward()
        expected=torch.ones_like(squared);expected[:,6:12]=2
        torch.testing.assert_close(squared.grad,expected)

    def test_invalid_weights_and_checkpoint_compatibility(self):
        for weights in ((1.,)*5, (0.,)*6,(-1.,1.,1.,1.,1.,1.),(float('nan'),)*6):
            with self.assertRaises(ValueError):self.model(weights)
        source=self.model();target=self.model((1.,1.,2.,2.,1.,1.))
        target.load_state_dict(source.state_dict(),strict=True)

    def test_block_logs_sum_to_weighted_state_loss(self):
        model=self.model((1.,1.,2.,2.,1.,1.))
        s=state();true=s.clone();true[:,2]+=.02;true[:,38]+=.1;true[:,41]+=.2
        action=torch.zeros(2,4,29);context=torch.zeros(4,35);context[:,0]=.75
        _,metrics=model.window_loss(torch.stack([s,true,true]),action,context,
                                    torch.zeros(2,4),[reference(),reference()])
        total=sum(metrics['state_loss_block/'+name] for name,_,_ in wm.STATE_ERROR_BLOCKS)
        torch.testing.assert_close(total,metrics['state_loss'])
        for name,_,_ in wm.STATE_ERROR_BLOCKS:
            for prefix in ('state_nse_first/','state_nse_last/','state_loss_block/'):
                self.assertFalse(metrics[prefix+name].requires_grad)


if __name__=='__main__':unittest.main()
