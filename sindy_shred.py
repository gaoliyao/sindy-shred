import torch
from torch.utils.data import DataLoader
import numpy as np
from sindy import sindy_library_torch, e_sindy_library_torch

class SINDy(torch.nn.Module):
    def __init__(self, latent_dim, library_dim, poly_order, include_sine):
        super().__init__()
        self.latent_dim = latent_dim
        self.poly_order = poly_order
        self.include_sine = include_sine
        self.library_dim = library_dim
        self.coefficients = torch.ones(library_dim, latent_dim, requires_grad=True)
        torch.nn.init.normal_(self.coefficients, mean=0.0, std=0.001)
        self.coefficient_mask = torch.ones(library_dim, latent_dim, requires_grad=False).cuda()
        self.coefficients = torch.nn.Parameter(self.coefficients)

    def forward(self, h, dt):
        library_Theta = sindy_library_torch(h, self.latent_dim, self.poly_order, self.include_sine)
        h = h + library_Theta @ (self.coefficients * self.coefficient_mask) * dt
        return h
    
    def thresholding(self, threshold):
        self.coefficient_mask = torch.abs(self.coefficients) > threshold
        self.coefficients.data = self.coefficient_mask * self.coefficients.data
        
    def add_noise(self, noise=0.1):
        self.coefficients.data += torch.randn_like(self.coefficients.data) * noise
        self.coefficient_mask = torch.ones(self.library_dim, self.latent_dim, requires_grad=False).cuda()   
        
    def recenter(self):
        self.coefficients.data = torch.randn_like(self.coefficients.data) * 0.0
        self.coefficient_mask = torch.ones(self.library_dim, self.latent_dim, requires_grad=False).cuda()   

class E_SINDy(torch.nn.Module):
    def __init__(self, num_replicates, latent_dim, library_dim, poly_order, include_sine):
        super().__init__()
        self.num_replicates = num_replicates
        self.latent_dim = latent_dim
        self.poly_order = poly_order
        self.include_sine = include_sine
        self.library_dim = library_dim
        self.coefficients = torch.ones(num_replicates, library_dim, latent_dim, requires_grad=True)
        torch.nn.init.normal_(self.coefficients, mean=0.0, std=0.001)
        self.coefficient_mask = torch.ones(num_replicates, library_dim, latent_dim, requires_grad=False).cuda()
        self.coefficients = torch.nn.Parameter(self.coefficients)

    def forward(self, h_replicates, dt):
        num_data, num_replicates, latent_dim = h_replicates.shape
        h_replicates = h_replicates.reshape(num_data * num_replicates, latent_dim)
        library_Thetas = e_sindy_library_torch(h_replicates, self.latent_dim, self.poly_order, self.include_sine)
        library_Thetas = library_Thetas.reshape(num_data, num_replicates, self.library_dim)
        h_replicates = h_replicates.reshape(num_data, num_replicates, latent_dim)
        h_replicates = h_replicates + torch.einsum('ijk,jkl->ijl', library_Thetas, (self.coefficients * self.coefficient_mask)) * dt
        return h_replicates
    
    def thresholding(self, threshold, base_threshold=0):
        threshold_tensor = torch.full_like(self.coefficients, threshold)
        for i in range(self.num_replicates):
            threshold_tensor[i] = threshold_tensor[i] * 10**(0.2 * i - 1) + base_threshold
        self.coefficient_mask = torch.abs(self.coefficients) > threshold_tensor
        self.coefficients.data = self.coefficient_mask * self.coefficients.data
        
    def add_noise(self, noise=0.1):
        self.coefficients.data += torch.randn_like(self.coefficients.data) * noise
        self.coefficient_mask = torch.ones(self.num_replicates, self.library_dim, self.latent_dim, requires_grad=False).cuda()   
        
    def recenter(self):
        self.coefficients.data = torch.randn_like(self.coefficients.data) * 0.0
        self.coefficient_mask = torch.ones(self.num_replicates, self.library_dim, self.latent_dim, requires_grad=False).cuda()  

class SINDy_SHRED(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64, hidden_layers=1, l1=350, l2=400, dropout=0.0, library_dim=10, poly_order=3, include_sine=False, dt=0.03, layer_norm=False):
        super(SINDy_SHRED, self).__init__()
        self.gru = torch.nn.GRU(input_size=input_size, hidden_size=hidden_size,
                                        num_layers=hidden_layers, batch_first=True).cuda()
        self.num_replicates = 10
        self.e_sindy = E_SINDy(self.num_replicates, hidden_size, library_dim, poly_order, include_sine).cuda()
        
        self.linear1 = torch.nn.Linear(hidden_size, l1)
        self.linear2 = torch.nn.Linear(l1, l2)
        self.linear3 = torch.nn.Linear(l2, output_size)

        self.dropout = torch.nn.Dropout(dropout)

        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.dt = dt
        self.use_layer_norm = layer_norm
        self.layer_norm_gru = torch.nn.LayerNorm(hidden_size)

    def forward(self, x, sindy=False):
        h_0 = torch.zeros((self.hidden_layers, x.size(0), self.hidden_size), dtype=torch.float)
        if next(self.parameters()).is_cuda:
            h_0 = h_0.cuda()
        
        _, h_out = self.gru(x, h_0)
        h_out = h_out[-1].view(-1, self.hidden_size)
        if self.use_layer_norm:
            h_out = self.layer_norm_gru(h_out)

        output = self.linear1(h_out)
        output = self.dropout(output)
        output = torch.nn.functional.relu(output)

        output = self.linear2(output)
        output = self.dropout(output)
        output = torch.nn.functional.relu(output)
    
        output = self.linear3(output)
        with torch.autograd.set_detect_anomaly(True):
            if sindy:
                h_t = h_out[:-1, :]
                ht_replicates = h_t.unsqueeze(1).repeat(1, self.num_replicates, 1)
                for _ in range(10):
                    ht_replicates = self.e_sindy(ht_replicates, dt=self.dt)
                h_out_replicates = h_out[1:, :].unsqueeze(1).repeat(1, self.num_replicates, 1)
                output = output, h_out_replicates, ht_replicates
        return output
    
    def gru_outputs(self, x, sindy=False):
        h_0 = torch.zeros((self.hidden_layers, x.size(0), self.hidden_size), dtype=torch.float)
        if next(self.parameters()).is_cuda:
            h_0 = h_0.cuda()
        _, h_out = self.gru(x, h_0)
        h_out = h_out[-1].view(-1, self.hidden_size)
        if self.use_layer_norm:
            h_out = self.layer_norm_gru(h_out)

        if sindy:
            h_t = h_out[:-1, :]
            ht_replicates = h_t.unsqueeze(1).repeat(1, self.num_replicates, 1)
            for _ in range(10):
                ht_replicates = self.e_sindy(ht_replicates, dt=self.dt)
            h_out_replicates = h_out[1:, :].unsqueeze(1).repeat(1, self.num_replicates, 1)
            h_outs = h_out_replicates, ht_replicates
        return h_outs
    
    def sindys_threshold(self, threshold):
        self.e_sindy.thresholding(threshold)
            
    def sindys_add_noise(self, noise):
        self.e_sindy.add_noise(noise)

def fit(model, train_dataset, valid_dataset, batch_size=64, num_epochs=4000, lr=1e-3, sindy_regularization=1.0, optimizer="AdamW", verbose=False, threshold=0.5, base_threshold=0.0, patience=20, thres_epoch=100, weight_decay=0.01):
    train_loader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size)
    criterion = torch.nn.MSELoss()
    if optimizer == "AdamW":
        optimizer = torch.optim.AdamW([{"params": model.gru.parameters(), "lr": 0.01},
                                       {"params": model.linear1.parameters()},
                                       {"params": model.linear2.parameters()},
                                       {"params": model.linear3.parameters()}], lr=lr, weight_decay=0.01)
        optimizer_sindy = torch.optim.AdamW([{"params": model.e_sindy.parameters(), "lr": 0.01}], lr=lr, weight_decay=1)
        optimizer_everything = torch.optim.AdamW([{"params": model.gru.parameters()},
                                                  {"params": model.linear1.parameters()},
                                                  {"params": model.linear2.parameters()},
                                                  {"params": model.linear3.parameters()},
                                                  {"params": model.e_sindy.parameters()}], lr=lr, weight_decay=0.01)
    
    val_error_list = []
    patience_counter = 0
    best_params = model.state_dict()
    for epoch in range(1, num_epochs + 1):
        for data in train_loader:
            model.train()
            outputs, h_gru, h_sindy = model(data[0], sindy=True)
            optimizer_everything.zero_grad()
            loss = criterion(outputs, data[1]) + criterion(h_gru, h_sindy) * sindy_regularization + torch.abs(torch.mean(h_gru)) * 0.1
            loss.backward()
            optimizer_everything.step()
        print(epoch, ":", loss)
        if epoch % thres_epoch == 0 and epoch != 0:
            model.e_sindy.thresholding(threshold=threshold, base_threshold=base_threshold)
            model.eval()
            with torch.no_grad():
                val_outputs = model(valid_dataset.X)
                val_error = torch.linalg.norm(val_outputs - valid_dataset.Y)
                val_error = val_error / torch.linalg.norm(valid_dataset.Y)
                val_error_list.append(val_error)
            if verbose:
                print('Training epoch ' + str(epoch))
                print('Error ' + str(val_error_list[-1]))
            if val_error == torch.min(torch.tensor(val_error_list)):
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter == patience:
                return torch.tensor(val_error_list).cpu()
    return torch.tensor(val_error_list).detach().cpu().numpy()

def forecast(forecaster, reconstructor, test_dataset):
    initial_in = test_dataset.X[0:1].clone()
    vals = [initial_in[0, i, :].detach().cpu().clone().numpy() for i in range(test_dataset.X.shape[1])]
    for i in range(len(test_dataset.X)):
        scaled_output1, scaled_output2 = forecaster(initial_in)
        scaled_output1 = scaled_output1.detach().cpu().numpy()
        scaled_output2 = scaled_output2.detach().cpu().numpy()
        vals.append(np.concatenate([scaled_output1.reshape(test_dataset.X.shape[2]//2), scaled_output2.reshape(test_dataset.X.shape[2]//2)]))
        temp = initial_in.clone()
        initial_in[0, :-1] = temp[0, 1:]
        initial_in[0, -1] = torch.tensor(np.concatenate([scaled_output1, scaled_output2]))
    device = 'cuda' if next(reconstructor.parameters()).is_cuda else 'cpu'
    forecasted_vals = torch.tensor(np.array(vals), dtype=torch.float32).to(device)
    reconstructions = []
    for i in range(len(forecasted_vals) - test_dataset.X.shape[1]):
        recon = reconstructor(forecasted_vals[i:i + test_dataset.X.shape[1]].reshape(1, test_dataset.X.shape[1], test_dataset.X.shape[2])).detach().cpu().numpy()
        reconstructions.append(recon)
    reconstructions = np.array(reconstructions)
    return forecasted_vals, reconstructions
