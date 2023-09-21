import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from datetime import datetime
import os
from VAE_model import VAE
import random
from torchsummary import summary


def plot_predictions_admission(net, data_loader, device):
    net.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device).view(-1, 1)
            _, _,_,_,_,outputs = net(inputs)
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())

    plt.scatter(y_true, y_pred)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.show()

def r_squared(y_true, y_pred):
    y_bar = np.mean(y_true)
    ss_tot = np.sum((y_true - y_bar) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


def plot_prediction_vs_truth_addmission(model, X_s, X, y,path,title="Model Predictions vs Truth", num_points=None,):
    device = next(model.parameters()).device
    fig, ax = plt.subplots()

    # Fetch CGPA values directly from DataFrame X
    cgpa_values = X.loc[:, 'x1'].values
    normalized_cgpa = (cgpa_values - np.min(cgpa_values)) / (np.max(cgpa_values) - np.min(cgpa_values))
    cmap = plt.get_cmap('coolwarm')  # Using 'coolwarm' colormap. You can choose any appropriate colormap.

    with torch.no_grad():
        model = model.eval()
        predictions = model(torch.tensor(X_s, dtype=torch.float, device=device)).cpu().numpy()

    # Ensure all arrays have the same length before looping
    assert len(y) == len(X_s) == len(normalized_cgpa), "Input arrays must have the same length"
    
    indices = range(len(X_s))
    
    # Sample num_points indices
    if num_points and num_points < len(indices):
        indices = random.sample(indices, num_points)
    
    for i in indices:
        color = cmap(normalized_cgpa[i])
        plt.scatter(y.iloc[i], predictions[i], color=color, alpha=0.7)

    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)

    plt.xlabel("Truth")
    plt.ylabel("Prediction")
    plt.title(title)
    
    sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    
    # Define the colorbar
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', label='x1')

    # Customize colorbar ticks
    min_val, max_val = np.min(cgpa_values), np.max(cgpa_values)
    tick_values = np.linspace(min_val, max_val, num=6)  # 6 ticks
    normed_ticks = (tick_values - min_val) / (max_val - min_val)  # Normalize tick values (0 to 1 range)
    cbar.set_ticks(normed_ticks)
    
    # Set tick labels with two decimal places
    cbar.set_ticklabels(['{:.2f}'.format(val) for val in tick_values])
    plt.savefig(path)
    #plt.show()



def vae_loss(recon_x, x, mu_z, logvar_z,mu_zz,logvar_zz,y_true,y_pred,b1,b2,b3,b4,epoch,printel=False,wabdb=False,partial=-1):
    MSE = nn.functional.mse_loss(recon_x, x, reduction='mean')
    pred_loss = torch.sqrt(nn.functional.mse_loss(y_true, y_pred, reduction='mean'))
    KLD_z = kl_divergence_loss(mu_z,logvar_z)
    KLD_zz =kl_divergence_loss(mu_zz,logvar_zz)
    z=reparameterize(mu_z, logvar_z,10)
    zz=reparameterize(mu_zz, logvar_zz,10) 

    if printel:
        print('MSE:',MSE)
        print('KLD_z:',KLD_z)
        print('KLD_zz:',KLD_zz)
        print('pred loss',pred_loss)
    
    if wabdb:
        wandb.log({"MSE": MSE, 'KLD_z:':KLD_z,'KLD_zz:':KLD_zz,'pred_loss:':pred_loss})
    

    if partial>0:
        if epoch<partial:
            return b1*MSE+ b2*KLD_z+b3*KLD_zz
        else:
            return b4*pred_loss

    
    return b1*MSE + b2*KLD_z+b3*KLD_zz+b4*pred_loss

def reparameterize( mu, logvar, n):
        samples = []
        for _ in range(n):
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            sample = mu + eps*std
            samples.append(sample)
        return torch.mean(torch.stack(samples), dim=0)

def kl_divergence_loss(mu, logvar):
    # This implementation assumes the target distribution is a standard normal distribution.
    # The KL divergence is then -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2).
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD

def create_picture_name(combined_config, prefix_string):
    config = combined_config['config']
    beta = combined_config['beta']
    lr = str(combined_config['lr']).replace(".", "")
    batch_size = combined_config['batch_size']
    num_epochs = combined_config['num_epochs']
    
    encoder_z_layers = '_'.join(map(str, config['encoder_z_layers']))
    encoder_zz_layers = '_'.join(map(str, config['encoder_zz_layers']))
    decoder_layers = '_'.join(map(str, config['decoder_layers']))
    fc_z_to_y_layers = '_'.join(map(str, config['fc_z_to_y_layers']))
    
    pic_name = f"{prefix_string}_VAE_xdim{config['x_dim']}_zdim{config['z_dim']}_zzdim{config['zz_dim']}_lr{lr}_bs{batch_size}_epochs{num_epochs}_b1{beta['b1']}_b2{beta['b2']}_b3{beta['b3']}_b4{beta['b4']}_b5{beta['b5']}_encZ{encoder_z_layers}_encZZ{encoder_zz_layers}_dec{decoder_layers}_fc{fc_z_to_y_layers}.png"
    
    return pic_name


def train_VAE(X_train,X_test,X_test_original,y_train,y_test,config,path,save_pictures=False,freeze_weights=True,wabdb=False,z_and_zz_to_y=True,printel=False,partial=-1,offset=200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if wabdb:
        run = wandb.init(project="VAE_TOY_zz_and_z_to_y_freeze", name=f'run-{"{:%Y_%m_%d_%H_%M_%S}".format(datetime.now())}')
        wandb.log(config)  # log all the configuration parameters

    # Extract parameters
    config_params = config['config']
    beta = config['beta']
    lr = config['lr']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    
    x_dim = config_params['x_dim']
    z_dim = config_params['z_dim']
    zz_dim = config_params['zz_dim']
    encoder_z_layers = config_params['encoder_z_layers']
    encoder_zz_layers = config_params['encoder_zz_layers']
    decoder_layers = config_params['decoder_layers']
    fc_z_to_y_layers = config_params['fc_z_to_y_layers']
    b1 = beta['b1']
    b2 = beta['b2']
    b3 = beta['b3']
    b4 = beta['b4']
    b5 = beta['b5']
    print(config)

    model = VAE(x_dim, z_dim, zz_dim, encoder_z_layers, encoder_zz_layers, decoder_layers, fc_z_to_y_layers,z_and_zz_to_y=z_and_zz_to_y)
    print(wabdb)
    if wabdb:
        wandb.watch(model)
    model.to(device)
    summary(model, input_size=(x_dim,), device=device.type)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # define your scheduler
    scheduler = StepLR(optimizer, step_size=8000, gamma=0.95)
    y_train_tensor = torch.from_numpy(y_train).float().to(device)  # convert y_train to tensor
    X_train_tensor = torch.from_numpy(X_train).float().to(device)
    data = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    test_ds = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    losses = []

    for epoch in range(num_epochs):

        if freeze_weights:
            if partial==-1:
                print("please specify partial value")
        
            if epoch==partial:
                for param in model.encoder_z_seq.parameters():
                        param.requires_grad = False
                #for param in model.decoder.parameters():
                        #param.requires_grad = False
                for param in model.fc21.parameters():
                        param.requires_grad=False
                for param in model.fc22.parameters():
                        param.requires_grad=False
                """
                for param in model.encoder_zz_seq.parameters():
                        param.requires_grad = False
                for param in model.fcz21.parameters():
                        param.requires_grad=False
                for param in model.fcz22.parameters():
                        param.requires_grad=False
                """
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
                    
            if epoch==partial+offset:
                for param in model.encoder_zz_seq.parameters():
                        param.requires_grad = False
                for param in model.fcz21.parameters():
                        param.requires_grad=False
                for param in model.fcz22.parameters():
                        param.requires_grad=False
                for param in model.decoder.parameters():
                        param.requires_grad = False
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))



        for batch_idx, (x_batch,y_batch) in enumerate(data_loader):
            optimizer.zero_grad()
            recon_batch,mu_z, logvar_z,mu_zz,logvar_zz, y_pred = model(x_batch)
            loss = vae_loss(recon_batch, x_batch, mu_z, logvar_z,mu_zz,logvar_zz,y_batch,y_pred,b1,b2,b3,b4,epoch,printel=printel,partial=partial)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            scheduler.step()
        if epoch%50==0:
            print(f'Epoch {epoch + 1}, loss: {loss.item()}')

    y_true = np.array([y.numpy() for _, y in test_ds],dtype=object)
    _,_,_,_,_,outputs = model(torch.Tensor(X_test).to(device))
    y_pred = outputs.cpu().detach().numpy()
    r2 = r2_score(y_true, y_pred)
    print("R^2 score:", r2)
    if wabdb:
        wandb.log({"R^2": r2})
        run.finish()

    #testloss mse prediction
    _,_,_,_,_,y_pred_test = model(torch.Tensor(X_test).to(device))
    _,_,_,_,_,y_pred_train = model(torch.from_numpy(X_train).float().to(device))
    y_pred_test = y_pred_test.cpu().detach()
    y_pred_train = y_pred_train.cpu().detach()          
    loss_test = torch.sqrt(nn.functional.mse_loss(torch.Tensor(y_test), y_pred_test, reduction='mean'))
    loss_train = torch.sqrt(nn.functional.mse_loss(torch.Tensor(y_train), y_pred_train, reduction='mean'))
    
    if save_pictures:
        
        # At the end of each epoch, save the loss plot
        plt.figure(figsize=(10,5))
        plt.title("Loss over time")
        plt.plot(losses)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.show()
    	
        # Construct the picture name and save the plot
        pic_name_loss = create_picture_name(config, "Loss_plot")
        #plt.savefig(os.path.join(path, pic_name_loss))

        r2_text = str(r2).replace(".", "_")
        #pic_name_tp = create_picture_name(config, f"R2_{r2_text}")
        pic_name_pred = create_picture_name(config, f"Pred_{r2_text}")
        plot_predictions_admission(model,test_loader,device)

        #plot_prediction_vs_truth_addmission(model, X_test, X_test_original, y_test,os.path.join(path, pic_name_tp))
        #pic_name_ice = create_picture_name(config, "ICE_plot")
        #temp_index = feature_names.index('temp')
        #ice_plot(model, X_test_scaled,X_test, y_test, temp_index, feature_names,'temp',os.path.join(my_path, pic_name_ice))
        # Pass the x_subset through the model
        recon_x_subset, mu_z, log_z, mu_zz, log_zz,_ = model(torch.from_numpy(X_test).float().to(device))
        # Reparameterize to get the estimated z values
        z_est_subset = model.reparameterize(mu_z, log_z,1)
        zz_est_subset = model.reparameterize(mu_zz, log_zz,1)


        z_columns = [f'z{i+1}' for i in range(z_dim)]
        zz_columns = [f'zz{i+1}' for i in range(zz_dim)]
        
        # Convert your tensor to a dataframe
        zz_est_df = pd.DataFrame(zz_est_subset.cpu().detach().numpy(), columns=zz_columns)
        z_est_df = pd.DataFrame(z_est_subset.cpu().detach().numpy(), columns=z_columns)
        # Reset the index of both dataframes
        X_test_reset = pd.DataFrame(X_test_original).reset_index(drop=True)
        z_est_df_reset = z_est_df.reset_index(drop=True)
        zz_est_df = zz_est_df.reset_index(drop=True)
        y_test_reset=  pd.DataFrame(y_test).reset_index(drop=True)
        print(y_test_reset)

        # Concatenate the dataframes
        df = pd.concat([X_test_reset, z_est_df_reset,zz_est_df,y_test_reset], axis=1)
        print(df.head(5))
        plt.figure(figsize=(15, 15))
        # Compute the correlation matrix
        corr = df.corr()

        # Generate a heatmap
        sns.heatmap(corr, cmap="YlGnBu",annot=True)
        # Save the figure
        pic_name_heat = create_picture_name(config, "heatmap")
        print(pic_name_heat)
        plt.savefig(os.path.join(path, pic_name_heat))
        #plt.show()
        

        

    return model,r2,loss_train,loss_test

