import torch
import torch.optim as optim


class UserAggregator:
    '''Aggregate per-user gradiant'''
    
    def __init__(self, args, model_cls, device):
        self.args = args
        self.model = model_cls(args).to(device)
        self.device = device

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self._init_grad_param_vecs()

    def _init_grad_param_vecs(self):
        self.user_grad = {}
        self.user_sample_num = None

        self.optimizer.zero_grad()

    def update(self, all_sample_num):
        self.update_model_grad(all_sample_num)
        self._init_grad_param_vecs()

    def update_model_grad(self, all_sample_num):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = torch.sum(
                    self.user_grad[name] / all_sample_num,
                    dim=0,
                ).cuda()
        self.optimizer.step()

    def collect(self, user_grad, user_sample_num):
        for name in user_grad:
            if name not in self.user_grad:
                self.user_grad[name] = user_grad[name]
            else:
                self.user_grad[name] += user_grad[name]

        if self.user_sample_num is None:
            self.user_sample_num = user_sample_num
        else:
            self.user_sample_num += user_sample_num

    def collect_by_uindex(self, user_grad, user_sample_num, uindex):
        assert len(self.user_grad) != 0, "collect_by_uindex cannot apply to empty user_grad!"
        for name in user_grad:
            self.user_grad[name][uindex] += user_grad[name]

        self.user_sample_num[uindex] += user_sample_num
