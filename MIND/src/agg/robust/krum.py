import torch
from agg.user import UserAggregator


class KrumAggregator(UserAggregator):
    def krum(self, user_grad, user_num):
        # flatten grad
        with torch.no_grad():
            user_flatten_grad = []
            for u_i in range(user_num):
                user_flatten_grad_i = []
                for name in user_grad:
                    user_flatten_grad_i.append(torch.flatten(user_grad[name][u_i]))
                user_flatten_grad_i = torch.cat(user_flatten_grad_i)
                user_flatten_grad.append(user_flatten_grad_i)
            user_flatten_grad = torch.stack(user_flatten_grad)

            # compute l2 distance between users
            user_scores = torch.zeros((user_num, user_num), device=user_flatten_grad.device)
            for u_i in range(user_num):
                user_scores[u_i] = torch.norm(
                    user_flatten_grad - user_flatten_grad[u_i],
                    dim=list(range(len(user_flatten_grad.shape)))[1:],
                )
                user_scores[u_i, u_i] = torch.inf

            # select summation od smallest n-f-2 scores
            topk_user_scores, _ = torch.topk(
                user_scores, k=user_num - self.args.krum_mal_num - 2, dim=1, largest=False
            )
            sm_user_scores = torch.sum(topk_user_scores, dim=1)

            # users with smallest score is selected as update gradient
            u_score, select_u = torch.topk(sm_user_scores, k=1, largest=False)
            select_u = select_u[0]
            
        agg_grad = {}
        for name in user_grad:
            agg_grad[name] = user_grad[name][select_u]
        return agg_grad

    def update_model_grad(self, all_sample_num):
        # Average by sample num
        for name in self.user_grad:
            self.user_grad[name] = (self.user_grad[name] / self.user_sample_num.view(
                -1, *((1,) * len(self.user_grad[name].shape[1:]))
            )).cuda()

        agg_grad = self.krum(self.user_grad, self.user_sample_num.shape[0])

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = agg_grad[name].cuda()
        self.optimizer.step()
