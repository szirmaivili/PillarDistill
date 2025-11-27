from torch.nn.utils import clip_grad

from .hook import Hook
import torch


class OptimizerHook(Hook):
    def __init__(self, grad_clip=None):
        self.grad_clip = grad_clip

    def clip_grads(self, params):
        clip_grad.clip_grad_norm_(
            filter(lambda p: p.requires_grad, params), **self.grad_clip
        )

    def after_train_iter(self, trainer):
        trainer.optimizer.zero_grad()
        # print(trainer.outputs["loss"])
        trainer.outputs["loss"].backward()
        if self.grad_clip is not None:
            self.clip_grads(trainer.model.parameters())
        trainer.optimizer.step()
        check_and_fix_weights(trainer.model)

# Ez új!
def check_and_fix_weights(model):
        has_nan_weights = False
        for name, param in model.named_parameters():
            if param.requires_grad and (torch.isnan(param).any() or torch.isinf(param).any()):
                #print(f"CRITICAL WARNING: NaN/Inf detected in weights of layer: {name}. Resetting/replacing.")
                has_nan_weights = True
                
                # Lehetőségek a NaN súlyok kezelésére:
                # 1. Helyreállítani a súlyokat az utolsó jó állapotból (ha elmentetted)
                #    Ez a legideálisabb, de bonyolultabb, mert ehhez el kellene menteni a model.state_dict-et.
                
                # 2. Helyettesíteni NaN/Inf súlyokat 0-val vagy egy kis random értékkel.
                #    Ez egy drasztikusabb lépés, de megakadályozza a terjedést.
                with torch.no_grad(): # Ezek a módosítások nem generálnak gradienst
                    param.data = torch.where(torch.isnan(param.data) | torch.isinf(param.data), 
                                             param.data.new_tensor(1e-5) * torch.randn_like(param.data), # Kis random érték
                                             param.data)
