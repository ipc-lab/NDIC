import argparse
import os
import numpy as np
import pandas as pd
import torch
import yaml
import PIL.Image as Image
from collections import OrderedDict
from torch.utils.data import DataLoader
from dataset.PairKitti import PairKitti
from models.balle2018.model import BMSHJ2018Model
from models.balle2017.model import BLS2017Model
from models.distributed_model import HyperPriorDistributedAutoEncoder, DistributedAutoEncoder
from pytorch_msssim import ms_ssim

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml', help="configuration")


def get_bpp(model_out, config):  # Returns calculated bpp for train and test
    alpha = config['alpha']
    beta = config['beta']
    if config['baseline_model'] == 'bmshj18':
        if config['use_side_info']:  # If the side information (correlated image) has to be used
            ''' 
            The loss function consists of:
            Rate terms for input image (likelihoods), correlated image (y_likelihoods),
            and the common information (w_likelihoods), hyperpriors for input image (z_likelihoods)
            , hyperpriors for correlated image (z_likelihoods_cor).
            Sum of these rate terms is returned as bpp, along with the actual bpp transmitted over the channel,
            which consists only of likelihoods + z_likelihoods.
            '''
            x_recon, y_recon, likelihoods, y_likelihoods, z_likelihoods, z_likelihoods_cor, w_likelihoods = model_out
            size_est = (-np.log(2) * x_recon.numel() / 3)
            bpp = (torch.sum(torch.log(likelihoods)) + torch.sum(torch.log(z_likelihoods))) / size_est
            transmitted_bpp = bpp.clone().detach()  # the real bpp value which is transmitted (for test)
            bpp += alpha * (torch.sum(torch.log(y_likelihoods)) + torch.sum(torch.log(z_likelihoods_cor))) / size_est
            bpp += beta * torch.sum(torch.log(w_likelihoods)) / size_est
            return bpp, transmitted_bpp
        else:  # The baseline implementation (Balle2018) without the side information
            x_recon, likelihoods, z_likelihoods = model_out
            size_est = (-np.log(2) * x_recon.numel() / 3)
            bpp = (torch.sum(torch.log(likelihoods)) + torch.sum(torch.log(z_likelihoods))) / size_est
            return bpp, bpp
    elif config['baseline_model'] == 'bls17':
        if config['use_side_info']:
            x_recon, y_recon, likelihoods, y_likelihoods, w_likelihoods = model_out
            size_est = (-np.log(2) * x_recon.numel() / 3)
            bpp = torch.sum(torch.log(likelihoods)) / size_est
            transmitted_bpp = bpp.clone().detach()  # the real bpp value which is transmitted (for test)
            bpp += alpha * torch.sum(torch.log(y_likelihoods)) / size_est
            bpp += beta * torch.sum(torch.log(w_likelihoods)) / size_est
            return bpp, transmitted_bpp
        else:
            x_recon, likelihoods = model_out
            size_est = (-np.log(2) * x_recon.numel() / 3)
            bpp = torch.sum(torch.log(likelihoods)) / size_est
            return bpp, bpp
    return None


def get_distortion(config, out, img, cor_img, mse):
    distortion = None
    alpha = config['alpha']
    if config['use_side_info']:
        ''' 
        The loss function consists of:
        Distortion terms for input image (x_recon), and correlated image (x_cor_recon).
        '''
        x_recon, y_recon = out[0], out[1]
        if config['distortion_loss'] == 'MS-SSIM':
            distortion = (1 - ms_ssim(img.cpu(), x_recon.cpu(), data_range=1.0, size_average=True,
                                      win_size=7))
            distortion += alpha * (1 - ms_ssim(cor_img.cpu(), y_recon.cpu(), data_range=1.0, size_average=True,
                                               win_size=7))
        elif config['distortion_loss'] == 'MSE':
            distortion = mse(img, x_recon)
            distortion += alpha * mse(cor_img, y_recon)
    else:
        x_recon = out[0]
        if config['distortion_loss'] == 'MS-SSIM':
            distortion = (1 - ms_ssim(img.cpu(), x_recon.cpu(), data_range=1.0, size_average=True,
                                      win_size=7))
        elif config['distortion_loss'] == 'MSE':
            distortion = mse(img, x_recon)

    return distortion

''' Since the pre-trained weights provided for bls17 by us were trained with
    different layer names, we map the layer names in the state dictionaries
    to the new names using the following function map_layers().
'''
def map_layers(weight):  
    return OrderedDict([(k.replace('z', 'w'), v) if 'z' in k else (k, v) for k, v in weight.items()])


def save_image(x_recon, x, path, name):
    img_recon = np.clip((x_recon * 255).squeeze().cpu().numpy(), 0, 255)
    img = np.clip((x * 255).squeeze().cpu().numpy(), 0, 255)
    img_recon = np.transpose(img_recon, (1, 2, 0)).astype('uint8')
    img = np.transpose(img, (1, 2, 0)).astype('uint8')
    img_final = Image.fromarray(np.concatenate((img, img_recon), axis=1), 'RGB')
    if not os.path.exists(path):
        os.makedirs(path)
    img_final.save(os.path.join(path, name + '.png'))


def main(config):
    # Dataset initialization
    path = config['dataset_path']
    resize = tuple(config['resize'])
    train_dataset = PairKitti(path=path, set_type='train', resize=resize)
    val_dataset = PairKitti(path=path, set_type='val', resize=resize)
    test_dataset = PairKitti(path=path, set_type='test', resize=resize)

    batch_size = config['train_batch_size']
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=3)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=3)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=3)

    # Model initialization
    '''
    We provide the option of two baseline models.
    1) The Balle2017 model (bls17).
    2) The Balle2018 model (bmshj18), which uses scale hyperpriors.
    '''
    with_side_info = config['use_side_info']
    model_class = None
    if config['baseline_model'] == 'bmshj18':
        if with_side_info:
            model_class = HyperPriorDistributedAutoEncoder
        else:
            model_class = BMSHJ2018Model
    elif config['baseline_model'] == 'bls17':
        if with_side_info:
            model_class = DistributedAutoEncoder
        else:
            model_class = BLS2017Model

    model = model_class(num_filters=config['num_filters'])
    model = model.cuda() if config['cuda'] else model
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], amsgrad=True)
    if config['load_weight']:
        checkpoint = torch.load(config['weight_path'], map_location=torch.device('cuda' if config['cuda'] else 'cpu'))
        if config['baseline_model'] == 'bls17' and with_side_info:
            checkpoint['model_state_dict'] = map_layers(checkpoint['model_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-7)
    experiment_name = model_class.__name__ + '_' + str(train_dataset) + '_' + config['distortion_loss'] + '_lambda:' + \
                      str(config['lambda'])

    print('Experiment: ', experiment_name)

    weight_folder = None
    if config['save_weights']:
        weight_folder = os.path.join(config['save_output_path'], 'weight')
        if not os.path.exists(weight_folder):
            os.makedirs(weight_folder)

    # Training initialization
    mse = torch.nn.MSELoss(reduction='mean')
    mse = mse.cuda() if config['cuda'] else mse
    lmbda = config['lambda']
    if config['train']:
        min_val_loss = None
        for epoch in range(config['epochs']):
            model.train()
            for i, data in enumerate(iter(train_loader)):
                img, cor_img, _, _ = data
                img = img.cuda().float() if config['cuda'] else img.float()
                cor_img = cor_img.cuda().float() if config['cuda'] else cor_img.float()

                optimizer.zero_grad()

                if with_side_info:
                    out = model(img, cor_img)
                else:
                    out = model(img)
                bpp, _ = get_bpp(out, config)

                distortion = get_distortion(config, out, img, cor_img, mse)

                loss = lmbda * distortion * (255 ** 2) + bpp  # multiplied by (255 ** 2) for distortion scaling
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            val_loss = []
            val_mse = []
            val_msssim = []
            val_bpp = []
            val_transmitted_bpp = []
            val_distortion = []
            with torch.no_grad():
                for i, data in enumerate(iter(val_loader)):
                    # img = input image, cor_img = side information/correlated image (designated y in the paper)
                    img, cor_img, _, _ = data
                    img = img.cuda().float() if config['cuda'] else img.float()
                    cor_img = cor_img.cuda().float() if config['cuda'] else cor_img.float()

                    if with_side_info:
                        out = model(img, cor_img)
                    else:
                        out = model(img)
                    bpp, transmitted_bpp = get_bpp(out, config)

                    x_recon = out[0]
                    mse_dist = mse(img, x_recon)
                    msssim = 1 - ms_ssim(img.clone().cpu(), x_recon.clone().cpu(), data_range=1.0, size_average=True,
                                         win_size=7)
                    msssim_db = -10 * np.log10(msssim)

                    distortion = get_distortion(config, out, img, cor_img, mse)

                    loss = lmbda * distortion * (255 ** 2) + bpp  # multiplied by (255 ** 2) for distortion scaling

                    val_mse.append(mse_dist.item())
                    val_bpp.append(bpp.item())
                    val_transmitted_bpp.append(transmitted_bpp.item())
                    val_loss.append(loss.item())
                    val_msssim.append(msssim_db.item())
                    val_distortion.append(distortion.item())

            val_loss_to_track = sum(val_loss) / len(val_loss)
            scheduler.step(val_loss_to_track)

            # Verbose
            if config['verbose_period'] > 0 and (epoch + 1) % config['verbose_period'] == 0:
                tracking = ['Epoch {}:'.format(epoch + 1),
                            'Loss = {:.4f},'.format(val_loss_to_track),
                            'BPP = {:.4f},'.format(sum(val_bpp) / len(val_bpp)),
                            'Distortion = {:.4f},'.format(sum(val_distortion) / len(val_distortion)),
                            'Transmitted BPP = {:.4f},'.format(sum(val_transmitted_bpp) / len(val_transmitted_bpp)),
                            'PSNR = {:.4f},'.format(10 * np.log10(1 / (sum(val_mse) / (len(val_mse))))),
                            'MS-SSIM = {:.4f}'.format(sum(val_msssim) / len(val_msssim))]
                print(" ".join(tracking))

            # Save weights
            if config['save_weights']:
                if min_val_loss is None or min_val_loss > val_loss_to_track:
                    min_val_loss = val_loss_to_track
                    save_path = os.path.join(weight_folder, experiment_name + '.pt')
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    }, save_path)

    if config['test']:
        results_path = os.path.join(config['save_output_path'], 'results')
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        names = ["Image Number", "BPP", "PSNR", "MS-SSIM"]
        cols = dict()
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(iter(test_loader)):
                img, cor_img, _, _ = data
                img = img.cuda().float() if config['cuda'] else img.float()
                cor_img = cor_img.cuda().float() if config['cuda'] else cor_img.float()

                if with_side_info:
                    out = model(img, cor_img)
                else:
                    out = model(img)
                bpp, transmitted_bpp = get_bpp(out, config)

                x_recon = out[0]
                mse_dist = mse(img, x_recon)
                msssim = 1 - ms_ssim(img.clone().cpu(), x_recon.clone().cpu(), data_range=1.0, size_average=True,
                                     win_size=7)
                msssim_db = -10 * np.log10(msssim)

                vals = [str(i)] + ['{:.8f}'.format(x) for x in [transmitted_bpp.item(),
                                                                10 * np.log10(1 / mse_dist.item()),
                                                                msssim_db.item()]]

                for (name, val) in zip(names, vals):
                    if name not in cols:
                        cols[name] = []
                    cols[name].append(val)

                if config['save_image']:
                    save_image(x_recon[0], img[0], os.path.join(results_path, '{}_images'.format(experiment_name)), str(i))

            df = pd.DataFrame.from_dict(cols)
            df.to_csv(os.path.join(results_path, experiment_name + '.csv'))


if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.config, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    main(config)
