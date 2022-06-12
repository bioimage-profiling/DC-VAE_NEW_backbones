from tqdm import tqdm
import numpy as np
from PIL import Image
import argparse

import torch
import pickle
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

import matplotlib.pyplot as plt
from models.models_high import *
from utils.eval_high import *
from utils.misc_high import *

parser = argparse.ArgumentParser(description='Progressive Growing of GANs')
parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
        help='Please specify path to the ImageNet training data.')
parser.add_argument('--output_dir', default='/out' , type=str,
        help='Please specify path to the ImageNet training data.')

parser.add_argument('-d', '--data', default='celeba', type=str,
                    choices=['celeba', 'lsun'],
                    help=('Specify dataset. '
                          'Currently CelebA and LSUN is supported'))
parser.add_argument('-load', '--Load', default=False, type=bool,
                    help=('load'))
parser.add_argument('-eph', '--epoch', default=0, type=int,
                    help=('load'))
#############################
# Hyperparameters
#############################
seed               = 123
'''
lr                 = 0.0002
beta1              = 0.0
beta2              = 0.9
'''
lr=0.001
beta1              = 0.0
beta2              = 0.99
num_workers        = 2
data_path          = "dataset"

dis_batch_size     = 64
gen_batch_size     = 128
max_epoch          = 800
lambda_kld         = 1e-6
latent_dim         = 512
cont_dim           = 16
cont_k             = 8192
cont_temp          = 0.07

n_label = 1
batch_size = 16
# multi-scale contrastive setting
layers             = ["0","-1"]
args = parser.parse_args()
print(args.output_dir)
name =("").join(layers)
log_fname = f"{args.output_dir}/logs/celeba-{name}"
fid_fname = f"{args.output_dir}/logs/FID_celeba-{name}"
viz_dir = f"{args.output_dir}/viz/celeba-{name}"
models_dir = f"{args.output_dir}/saved_models/celeba-{name}"
if not os.path.exists(args.output_dir+"/logs"):
    os.makedirs(args.output_dir+"/logs")
if not os.path.exists(viz_dir):
    os.makedirs(viz_dir)
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
lambda_cont = 1.0/len(layers)
fix_seed(random_seed=seed)
#############################
# Make and initialize the Networks
#############################
encoder = torch.nn.DataParallel(Encoder(latent_dim,n_label)).cuda()
decoder = torch.nn.DataParallel(Decoder(latent_dim-n_label,n_label)).cuda()
dual_encoder = torch.nn.DataParallel(DualEncoder(cont_dim,n_label)).cuda()
dual_encoder_M = torch.nn.DataParallel(DualEncoder(cont_dim,n_label)).cuda()
for p, p_momentum in zip(dual_encoder.parameters(), dual_encoder_M.parameters()):
    p_momentum.data.copy_(p.data)
    p_momentum.requires_grad = False
gen_avg_param = copy_params(decoder)
d_queue, d_queue_ptr = {}, {}
for layer in layers:
    d_queue[layer] = torch.randn(cont_dim, cont_k).cuda()
    d_queue[layer] = F.normalize(d_queue[layer], dim=0)
    d_queue_ptr[layer] = torch.zeros(1, dtype=torch.long)

#############################
# Make the optimizers
#############################
opt_encoder = torch.optim.Adam(filter(lambda p: p.requires_grad, 
                                        encoder.parameters()),
                                lr, (beta1, beta2))
opt_decoder = torch.optim.Adam(filter(lambda p: p.requires_grad, 
                                        decoder.parameters()),
                                lr, (beta1, beta2))
shared_params = list(dual_encoder.module.progression.parameters()) + \
                list(dual_encoder.module.from_rgb.parameters())
opt_shared = torch.optim.Adam(filter(lambda p: p.requires_grad, 
                                        shared_params),
                                lr, (beta1, beta2))
opt_disc_head = torch.optim.Adam(filter(lambda p: p.requires_grad, 
                    dual_encoder.module.linear.parameters()),
                lr, (beta1, beta2))
cont_params = list(dual_encoder.module.head.parameters()) 

opt_cont_head = torch.optim.Adam(filter(lambda p: p.requires_grad, cont_params),
                    lr, (beta1, beta2))
#############################
# Load weights 
#############################


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def lsun_loader(path):
    def loader(transform):
        data = datasets.LSUNClass(
            path, transform=transform,
            target_transform=lambda x: 0)
        data_loader = DataLoader(data, shuffle=False, batch_size=batch_size,
                                 num_workers=4)

        return data_loader

    return loader


def celeba_loader(path):
    def loader(transform):

        data = datasets.ImageFolder(path, transform=transform)
        data_loader = DataLoader(data, shuffle=True, batch_size=batch_size,
                                 num_workers=4)

        return data_loader

    return loader


def sample_data(dataloader, image_size=4):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    loader = dataloader(transform)

    for img, label in loader:
        yield img, label


def train(encoder, decoder,dual_encoder,dual_encoder_M,gen_avg_param, loader):
    global_steps=0
    step = 0
    alpha = 0
    iteration = 0
    stabilize = False
    if args.Load:
        print("yreee")
        encoder.load_state_dict(torch.load(os.path.join(models_dir, f"{args.epoch}_encoder.sd")))
        decoder.load_state_dict(torch.load(os.path.join(models_dir, f"{args.epoch}_decoder_avg.sd")))
        gen_avg_param=copy_params(decoder)
        decoder.load_state_dict(torch.load(os.path.join(models_dir, f"{args.epoch}_decoder.sd")))
        dual_encoder.load_state_dict(torch.load(os.path.join(models_dir, f"{args.epoch}_dual_encoder.sd")))
        dual_encoder_M.load_state_dict(torch.load(os.path.join(models_dir, f"{args.epoch}_dual_encoder_M.sd")))
        opt_encoder.load_state_dict(torch.load(os.path.join(models_dir, f"{args.epoch}_opt_encoder.sd")))
        opt_decoder.load_state_dict(torch.load(os.path.join(models_dir, f"{args.epoch}_opt_decoder.sd")))
        opt_shared.load_state_dict(torch.load(os.path.join(models_dir, f"{args.epoch}_opt_shared.sd")))
        opt_cont_head.load_state_dict(torch.load(os.path.join(models_dir, f"{args.epoch}_opt_cont_head.sd")))
        opt_disc_head.load_state_dict(torch.load(os.path.join(models_dir, f"{args.epoch}_opt_disc_head.sd")))
        for layer in layers:
            d_queue[layer]=torch.load(os.path.join(models_dir, f"{args.epoch}_{layer}_queue.sd"))
            d_queue_ptr[layer]=torch.load(os.path.join(models_dir, f"{args.epoch}_{layer}_queueptr.sd"))
        with open(os.path.join(models_dir, f"{args.epoch}_status.pkl"), 'rb') as fp:
            dict_ = pickle.load(fp)
            global_steps=dict_["global_steps"]
            print("global")
            print(global_steps)
            step = 0
            alpha = dict_["alpha"]
            print(alpha)
            iteration = dict_["iteration"]
            iteration=90000
            print(iteration)

            stabilize = dict_["stabilize"]
            print(stabilize)
    print(step)
    encoder.train()
    decoder.train()
    dual_encoder.train()
    dataset = sample_data(loader, 4 * 2 ** step)
    pbar = tqdm(range(600000-global_steps))

    for i in pbar:

        alpha = min(1, 0.00002 * iteration)

        if stabilize is False and iteration > 50000:
            dataset = sample_data(loader, 4 * 2 ** step)
            stabilize = True

        if iteration > 100000:
            alpha = 0
            iteration = 0
            step += 1
            stabilize = False
            if step > 5:
                alpha = 1
                step = 5
            dataset = sample_data(loader, 4 * 2 ** step)

        try:
            real_image, label = next(dataset)

        except (OSError, StopIteration):
            dataset = sample_data(loader, 4 * 2 ** step)
            real_image, label = next(dataset)
        if batch_size!= real_image.shape[0]:
            continue
        curr_bs = real_image.shape[0]
        curr_log = f"{i+global_steps}:{i+global_steps}\t"
        real_imgs = real_image.type(torch.cuda.FloatTensor)
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (real_imgs.shape[0], latent_dim-n_label)))
        # ---------------------
        #  Train Discriminator
        # ---------------------
        opt_shared.zero_grad()
        opt_disc_head.zero_grad()
        real_validity,_ = dual_encoder(real_imgs, step, alpha, mode="dis")
        fake_imgs = decoder(z,label, step, alpha).detach()
        fake_validity,_ = dual_encoder(fake_imgs, step, alpha, mode="dis")
        rec, mu, logvar = f_recon(real_imgs, encoder, decoder, latent_dim-n_label,label,step,alpha)
        rec_validity,_ = dual_encoder(rec, step, alpha, mode="dis")
        # cal loss
        d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                 torch.mean(nn.ReLU(inplace=True)(1.0 + fake_validity))*0.5 + \
                 torch.mean(nn.ReLU(inplace=True)(1.0 + rec_validity))*0.5
        d_loss.backward()
        curr_log += f"d:{d_loss.item():.2f}\t"
        opt_shared.step()
        opt_disc_head.step()
        # -----------------
        #  Train Generator
        # -----------------
        if global_steps % 5 == 0:
            opt_decoder.zero_grad()
            opt_encoder.zero_grad()
            gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (real_imgs.shape[0], latent_dim-n_label)))
            gen_imgs = decoder(gen_z,label, step, alpha)
            fake_validity,_ = dual_encoder(gen_imgs, step, alpha, mode="dis")
            rec, mu, logvar = f_recon(real_imgs, encoder, decoder, latent_dim-n_label,label,step,alpha)
            rec_validity,_ = dual_encoder(rec, step, alpha, mode="dis")
            # cal loss
            g_loss = -(torch.mean(fake_validity)*0.5 + torch.mean(rec_validity)*0.5)
            kld = (-0.5 * torch.sum(1+logvar-mu.pow(2)-logvar.exp()))*lambda_kld
            (g_loss+kld).backward()
            opt_decoder.step()
            opt_encoder.step()
            curr_log += f"g:{g_loss.item():.2f}\t"
            # contrastive
            opt_encoder.zero_grad()
            opt_decoder.zero_grad()
            opt_shared.zero_grad()
            opt_cont_head.zero_grad()
            rec, mu, logvar = f_recon(real_imgs, encoder, decoder, latent_dim-n_label,label,step,alpha)
            im_k = real_imgs
            im_q = rec
            with torch.no_grad():
                # update momentum encoder
                for p, p_mom in zip(dual_encoder.parameters(), dual_encoder_M.parameters()):
                    p_mom.data = (p_mom.data*0.999) + (p.data*(1.0-0.999))
                d_k = dual_encoder_M(im_k, step, alpha, mode="cont")
                for l in layers:
                    d_k[l] = F.normalize(d_k[l], dim=1)
            total_cont = torch.tensor(0.0).cuda()
            d_q = dual_encoder(im_q, step, alpha, mode="cont")
            for l in layers:
                q = F.normalize(d_q[l], dim=1)
                k = d_k[l]
                queue = d_queue[l]
                l_pos = torch.einsum("nc,nc->n", [k,q]).unsqueeze(-1)
                l_neg = torch.einsum('nc,ck->nk', [q,queue.detach()])
                logits = torch.cat([l_pos, l_neg], dim=1) / cont_temp#0.07
                labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
                cont_loss = nn.CrossEntropyLoss()(logits, labels) * lambda_cont
                total_cont += cont_loss
                acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
                curr_log += f"cont{l}:{cont_loss.item():.1f}\t"
                curr_log += f"acc1{l}:{acc1.item():.1f}\t"
                curr_log += f"acc5{l}:{acc5.item():.1f}\t"
            kld = (-0.5 * torch.sum(1+logvar-mu.pow(2)-logvar.exp()))*lambda_kld
            (total_cont+kld).backward()
            opt_encoder.step()
            opt_decoder.step()
            opt_shared.step()
            opt_cont_head.step()

            for l in layers:
                ptr = int(d_queue_ptr[l])
                d_queue[l][:, ptr:(ptr+curr_bs)] = d_k[l].transpose(0,1)
                ptr = (ptr+curr_bs)%cont_k # move the pointer ahead
                d_queue_ptr[l][0] = ptr
            with torch.no_grad():
                rec_pix = torch.nn.MSELoss()(im_q, im_k).mean()
            # moving average weight
            for p, avg_p in zip(decoder.parameters(), gen_avg_param):
                avg_p.mul_(0.999).add_(0.001, p.data)
        
        if (global_steps+i)%500 == 0:
            print_and_save(curr_log, log_fname)
            viz_img = im_k[0:8].view(8,im_k.shape[1],im_k.shape[2],im_k.shape[3])
            viz_rec = im_q[0:].view(curr_bs,im_q.shape[1],im_q.shape[2],im_q.shape[3])
            out = torch.cat((viz_img, viz_rec), dim=0)
            fname = os.path.join(viz_dir, f"{i+global_steps}_recon.png")
            disp_images(out, fname, 8, norm="0.5")
            fname = os.path.join(viz_dir, f"{i+global_steps}_sample.png")
            disp_images(fake_imgs.view(-1,fake_imgs.shape[1],fake_imgs.shape[2],fake_imgs.shape[3]), fname, 8, norm="0.5")
        print(i+global_steps)
        
        if (i+global_steps) % 10000 == 0:
            decoder.eval()
            encoder.eval()
            backup_param = copy_params(decoder)
            load_params(decoder, gen_avg_param)
            #fid_sample = compute_fid_sample(decoder, latent_dim)
            #fid_recon = compute_fid_recon(encoder, decoder, test_loader, latent_dim)
            #S = f"epoch:{epoch} sample:{fid_sample} recon:{fid_recon}"
            # print_and_save(S, fid_fname)
            # save checkpoints
            torch.save(encoder.state_dict(), os.path.join(models_dir, f"{i+global_steps}_encoder.sd"))
            torch.save(decoder.state_dict(), os.path.join(models_dir, f"{i+global_steps}_decoder_avg.sd"))
            load_params(decoder, backup_param)
            torch.save(decoder.state_dict(), os.path.join(models_dir, f"{i+global_steps}_decoder.sd"))
            torch.save(dual_encoder.state_dict(), os.path.join(models_dir, f"{i+global_steps}_dual_encoder.sd"))
            torch.save(dual_encoder_M.state_dict(), os.path.join(models_dir, f"{i+global_steps}_dual_encoder_M.sd"))
            torch.save(opt_encoder.state_dict(), os.path.join(models_dir, f"{i+global_steps}_opt_encoder.sd"))
            torch.save(opt_decoder.state_dict(), os.path.join(models_dir, f"{i+global_steps}_opt_decoder.sd"))
            torch.save(opt_shared.state_dict(), os.path.join(models_dir, f"{i+global_steps}_opt_shared.sd"))
            torch.save(opt_cont_head.state_dict(), os.path.join(models_dir, f"{i+global_steps}_opt_cont_head.sd"))
            torch.save(opt_disc_head.state_dict(), os.path.join(models_dir, f"{i+global_steps}_opt_disc_head.sd"))
            for layer in layers:
                torch.save(d_queue[layer], os.path.join(models_dir, f"{i+global_steps}_{layer}_queue.sd"))
                torch.save(d_queue_ptr[layer], os.path.join(models_dir, f"{i+global_steps}_{layer}_queueptr.sd"))
            encoder.train()
            decoder.train()
            dictionary={"step":step,"alpha":alpha,"global_steps":i+global_steps,"iteration":iteration,"stabilize":stabilize}
            with open(os.path.join(models_dir, f"{i+global_steps}_status.pkl"), 'wb') as f:
                 pickle.dump(dictionary, f)

        
        iteration+=1
        print(iteration)
        '''
        if (i + 1) % 100 == 0:
            images = []
            for _ in range(5):
                input_class = Variable(torch.zeros(10).long()).cuda()
                images.append(g_running(
                    Variable(torch.randn(n_label * 10, code_size)).cuda(),
                    input_class, step, alpha).data.cpu())
            utils.save_image(
                torch.cat(images, 0),
                f'sample/{str(i + 1).zfill(6)}.png',
                nrow=n_label * 10,
                normalize=True,
                range=(-1, 1))

        if (i + 1) % 10000 == 0:
            torch.save(g_running, f'checkpoint/{str(i + 1).zfill(6)}.model')

        pbar.set_description(
            (f'{i + 1}; G: {gen_loss_val:.5f}; D: {disc_loss_val:.5f};'
             f' Grad: {grad_loss_val:.5f}; Alpha: {alpha:.3f}'))
        
        '''
if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    if args.data == 'celeba':
        loader = celeba_loader(args.data_path)

    elif args.data == 'lsun':
        loader = lsun_loader(args.data_path)

    train(encoder, decoder,dual_encoder,dual_encoder_M,gen_avg_param, loader)
