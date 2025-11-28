class Linear1(nn.Module):
    def __init__(self, dim_in, dim_mid, dim_out, k = 5):
        super(Linear1, self).__init__()
        self.fc = nn.ModuleList([nn.Sequential(nn.Linear(dim_in, dim_out)) for i in range(k)])
        self.num_layer = k
    def forward(self, tokens):
        
        token_list = []
        for i in range(self.num_layer):
            tokens = self.fc[i](tokens)
            token_list.append(tokens)
        return torch.cat(token_list, dim = 1)
