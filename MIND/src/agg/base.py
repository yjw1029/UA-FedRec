import torch.optim as optim

class BaseAggregator:
    '''Aggregate batch summed gradiant'''

    def __init__(self, args, model_cls, device):
        self.args = args
        self.model = model_cls(args).to(device)
        self.device = device

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self._init_grad_param_vecs()
    
    def _init_grad_param_vecs(self):
        self.optimizer.zero_grad()
    
    def update(self):
        self.update_model_grad()
        self._init_grad_param_vecs()
        
    def update_model_grad(self):
        self.optimizer.step()

    def collect(self, model_grad):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    param.grad = model_grad[name]
                else:
                    param.grad += model_grad[name]