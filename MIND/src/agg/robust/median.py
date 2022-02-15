import torch
from agg.user import UserAggregator


class MedianAggregator(UserAggregator):
    def update_model_grad(self, all_sample_num):
        # Average by sample num
        for name in self.user_grad:
            self.user_grad[name] = self.user_grad[name] / self.user_sample_num.view(
                -1, *((1,) * len(self.user_grad[name].shape[1:]))
            ).expand(-1, *self.user_grad[name].shape[1:])

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = torch.median(
                    self.user_grad[name],
                    dim=0,
                ).values.cuda()
        self.optimizer.step()
