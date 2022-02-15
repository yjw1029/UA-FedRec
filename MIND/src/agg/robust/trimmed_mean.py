import torch
from agg.user import UserAggregator


class TrimmedMeanAggregator(UserAggregator):
    @staticmethod
    def trimmed_mean(tensor, k, dim=0):
        m = tensor.size(dim)
        largest_value, _ = torch.topk(tensor, k=k, dim=dim)
        smallest_value, _ = torch.topk(tensor, k=k, dim=dim, largest=False)

        result = (
            torch.sum(tensor, dim=dim)
            - torch.sum(largest_value, dim=dim)
            - torch.sum(smallest_value, dim=dim)
        ) / (m - 2 * k)
        return result

    def update_model_grad(self, all_sample_num):
        # Average by sample num
        for name in self.user_grad:
            self.user_grad[name] = self.user_grad[name] / self.user_sample_num.view(
                -1, *((1,) * len(self.user_grad[name].shape[1:]))
            ).expand(-1, *self.user_grad[name].shape[1:])

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.trimmed_mean(
                    self.user_grad[name],
                    k=self.args.trimmed_mean_beta,
                    dim=0,
                ).cuda()
        self.optimizer.step()
