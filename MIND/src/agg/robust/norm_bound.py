import torch
from agg.user import UserAggregator


class NormBoundAggregator(UserAggregator):
    @staticmethod
    def clip_norm(user_grad, max_norm):
        total_norm = torch.norm(
            torch.stack(
                [
                    torch.norm(g, p=2, dim=list(range(len(g.shape)))[1:])
                    for g in user_grad.values()
                ]
            ),
            p=2,
            dim=0,
        )

        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)

        for name, p in user_grad.items():
            user_grad[name] = p * clip_coef_clamped.view(
                -1, *((1,) * len(user_grad[name].shape[1:]))
            )

        return user_grad

    def update_model_grad(self, all_sample_num):
        # Average by sample num
        for name in self.user_grad:
            self.user_grad[name] = self.user_grad[name] / self.user_sample_num.view(
                -1, *((1,) * len(self.user_grad[name].shape[1:]))
            )

        # clip norm
        self.user_grad = self.clip_norm(self.user_grad, max_norm=self.args.norm_bound)

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = torch.sum(
                    self.user_grad[name]
                    * self.user_sample_num.view(
                        -1, *((1,) * len(self.user_grad[name].shape[1:]))
                    )
                    / all_sample_num,
                    dim=0,
                ).cuda()
        self.optimizer.step()
