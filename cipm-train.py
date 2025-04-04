import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import scipy.io as sio
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import platform
from argparse import ArgumentParser
import types
from mambaIR import VSSBlock
# from mambaIR import VSSBlock,PatchEmbed,PatchUnEmbed
from dwconv import DEPTHWISECONV
from demo import PatchEmbed,PatchUnEmbed
from CCM import CCM
from CFM import Causal_Frequency_Mamba

parser = ArgumentParser(description='CIPM-Net')

parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=50, help='epoch number of end training')
parser.add_argument('--layer_num', type=int, default=9, help='phase number of CIPM-Net-plus')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--group_num', type=int, default=1, help='group number for training')
parser.add_argument('--cs_ratio', type=int, default=50, help='from {10, 20, 30, 40, 50}')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')

parser.add_argument('--matrix_dir', type=str, default='sampling_matrix', help='sampling matrix directory')
parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')

args = parser.parse_args()


start_epoch = args.start_epoch
end_epoch = args.end_epoch
learning_rate = args.learning_rate
layer_num = args.layer_num
group_num = args.group_num
cs_ratio = args.cs_ratio
gpu_list = args.gpu_list


try:
    # The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
    torch.backends.cuda.matmul.allow_tf32 = False
    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = False
except:
    pass


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


nrtrain = 800   # number of training blocks
batch_size = 16


# Load CS Sampling Matrix: phi
Phi_data_Name = './%s/mask_%d.mat' % (args.matrix_dir, cs_ratio)
Phi_data = sio.loadmat(Phi_data_Name)
mask_matrix = Phi_data['mask_matrix']


mask_matrix = torch.from_numpy(mask_matrix).type(torch.FloatTensor)
mask = torch.unsqueeze(mask_matrix, 2)
mask = torch.cat([mask, mask], 2)
mask = mask.to(device)


Training_data_Name = '' # Dataset
Training_data = sio.loadmat('./%s/%s' % (args.data_dir, Training_data_Name))
Training_labels = Training_data['labels']


if isinstance(torch.fft, types.ModuleType):
    class FFT_Mask_ForBack(torch.nn.Module):
        def __init__(self):
            super(FFT_Mask_ForBack, self).__init__()

        def forward(self, x, full_mask):
            full_mask = full_mask[..., 0]
            x_in_k_space = torch.fft.fft2(x)
            masked_x_in_k_space = x_in_k_space * full_mask.view(1, 1, *(full_mask.shape))
            masked_x = torch.real(torch.fft.ifft2(masked_x_in_k_space))
            return masked_x
else:
    class FFT_Mask_ForBack(torch.nn.Module):
        def __init__(self):
            super(FFT_Mask_ForBack, self).__init__()

        def forward(self, x, mask):
            x_dim_0 = x.shape[0]
            x_dim_1 = x.shape[1]
            x_dim_2 = x.shape[2]
            x_dim_3 = x.shape[3]
            x = x.view(-1, x_dim_2, x_dim_3, 1)
            y = torch.zeros_like(x)
            z = torch.cat([x, y], 3)
            fftz = torch.fft(z, 2)
            z_hat = torch.ifft(fftz * mask, 2)
            x = z_hat[:, :, :, 0:1]
            x = x.view(x_dim_0, x_dim_1, x_dim_2, x_dim_3)
            return x


# Define CIPM-Net-plus Block
class BasicBlock(torch.nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))


        self.conv_D = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1, 3, 3)))
        self.conv_D1 = nn.Parameter(init.xavier_normal_(torch.Tensor(16, 1, 3, 3)))
        self.conv_D2 = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 2, 3, 3)))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))

        self.conv_G = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 32, 3, 3)))
        self.VSS= VSSBlock(hidden_dim=32,drop_path=0.1,attn_drop_rate=0.1,d_state=2,is_light_sr=False)
        # self.VSS= VSSBlock(hidden_dim=32,drop_path=0.1,attn_drop_rate=0.1,d_state=1,is_light_sr=False)

        self.Dwconv = DEPTHWISECONV(in_ch=1, out_ch=1)
        self.patch=PatchEmbed()
        self.patch_embed = PatchUnEmbed()
        self.CCM = CCM()
        self.CFM = Causal_Frequency_Mamba()

    def forward(self, x, fft_forback, PhiTb, mask):
        x = x - self.lambda_step * fft_forback(x, mask)
        x = x + self.lambda_step * PhiTb
        x_input = x
        x_input = self.CCM(x_input)
        x_input = self.CSM(x_input)
        x_F = self.CFM(x)
        x_C = torch.cat((x_F, x_input), dim=1)
        x_C = x_C.sum(dim=1, keepdim=True)
        x_input = x + x_C
        x_D = F.conv2d(x_input, self.conv_D, padding=1)
        x = F.conv2d(x_D, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)
        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))
        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv2_backward, padding=1)
        x_G = F.conv2d(x_backward, self.conv_G, padding=1)
        x_pred = x_input + x_G
        x = F.conv2d(x_forward, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_D_est = F.conv2d(x, self.conv2_backward, padding=1)
        symloss = x_D_est - x_D

        return [x_pred, symloss]


# Define CIPM-Net-plus
class CIPM(torch.nn.Module):
    def __init__(self, LayerNo):
        super(CIPM, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo
        self.fft_forback = FFT_Mask_ForBack()

        for i in range(LayerNo):
            onelayer.append(BasicBlock())

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, PhiTb, mask):

        x = PhiTb

        layers_sym = []   # for computing symmetric loss

        for i in range(self.LayerNo):
            [x, layer_sym] = self.fcs[i](x, self.fft_forback, PhiTb, mask)
            layers_sym.append(layer_sym)

        x_final = x

        return [x_final, layers_sym]


model = CIPM(layer_num)
model = nn.DataParallel(model)
model = model.to(device)


print_flag = 1   # print parameter number

if print_flag:
    num_count = 0
    for para in model.parameters():
        num_count += 1
        print('Layer %d' % num_count)
        print(para.size())



class RandomDataset(Dataset):
    def __init__(self, data, length):
        self.data = data
        self.len = length

    def __getitem__(self, index):
        return torch.Tensor(self.data[index, :]).float()

    def __len__(self):
        return self.len

class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss

if (platform.system() =="Windows"):
    rand_loader = DataLoader(dataset=RandomDataset(Training_labels, nrtrain), batch_size=batch_size, num_workers=0,
                             shuffle=True)
else:
    rand_loader = DataLoader(dataset=RandomDataset(Training_labels, nrtrain), batch_size=batch_size, num_workers=4,
                             shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model_dir = "./%s/MRI_CS_CIPM_Net_plus_layer_%d_group_%d_ratio_%d" % (args.model_dir, layer_num, group_num, cs_ratio)

log_file_name = "./%s/Log_MRI_CS_CIPM_Net_plus_layer_%d_group_%d_ratio_%d.txt" % (args.log_dir, layer_num, group_num, cs_ratio)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)


if start_epoch > 0:
    pre_model_dir = model_dir
    model.load_state_dict(torch.load('./%s/CIPMnet_params_%d.pkl' % (pre_model_dir, start_epoch)))

charbonnier_loss = L1_Charbonnier_loss().to(device)

for epoch_i in range(start_epoch+1, end_epoch+1):
    for data in rand_loader:
        batch_x = data
        batch_x = batch_x.to(device)
        batch_x = batch_x.view(batch_x.shape[0], 1, batch_x.shape[1], batch_x.shape[2])

        PhiTb = FFT_Mask_ForBack()(batch_x, mask)

        [x_output, loss_layers_sym] = model(PhiTb, mask)

        # Compute and print loss
        loss_rec = torch.mean(torch.pow(x_output - batch_x, 2))
        loss_charbonnier = charbonnier_loss(x_output, batch_x)  # Charbonnier loss

        loss_pha = torch.mean(torch.pow(loss_layers_sym[0], 2))
        for k in range(layer_num-1):
            loss_pha += torch.mean(torch.pow(loss_layers_sym[k+1], 2))

        gamma = torch.Tensor([0.01]).to(device)
        alpha = torch.Tensor([0.1]).to(device)  # Charbonnier loss

        #  Charbonnier loss
        loss_all = loss_rec + torch.mul(gamma, loss_pha) + torch.mul(alpha, loss_charbonnier)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()

        output_data = "[%02d/%02d] Total Loss: %.5f, Rec Loss: %.5f, Pha Loss: %.5f, Charbonnier Loss: %.5f\n" % (
            epoch_i, end_epoch, loss_all.item(), loss_rec.item(), loss_pha.item(), loss_charbonnier.item()
        )
        print(output_data)

    with open(log_file_name, 'a') as output_file:
        output_file.write(output_data)

    if epoch_i % 1 == 0:
        torch.save(model.state_dict(), "./%s/net_params_%d.pkl" % (model_dir, epoch_i))  # save only the parameters


# Training loop
# for epoch_i in range(start_epoch+1, end_epoch+1):
#     for data in rand_loader:
#
#         batch_x = data
#         batch_x = batch_x.to(device)
#         batch_x = batch_x.view(batch_x.shape[0], 1, batch_x.shape[1], batch_x.shape[2])
#
#         PhiTb = FFT_Mask_ForBack()(batch_x, mask)
#
#         [x_output, loss_layers_sym] = model(PhiTb, mask)
#
#         # Compute and print loss
#         loss_discrepancy = torch.mean(torch.pow(x_output - batch_x, 2))
#
#         loss_constraint = torch.mean(torch.pow(loss_layers_sym[0], 2))
#         for k in range(layer_num-1):
#             loss_constraint += torch.mean(torch.pow(loss_layers_sym[k+1], 2))
#
#         gamma = torch.Tensor([0.01]).to(device)
#
#         # loss_all = loss_discrepancy
#         loss_all = loss_discrepancy + torch.mul(gamma, loss_constraint)
#
#         # Zero gradients, perform a backward pass, and update the weights.
#         optimizer.zero_grad()
#         loss_all.backward()
#         optimizer.step()
#
#         output_data = "[%02d/%02d] Total Loss: %.5f, Discrepancy Loss: %.5f,  Constraint Loss: %.5f\n" % (epoch_i, end_epoch, loss_all.item(), loss_discrepancy.item(), loss_constraint)
#         print(output_data)
#
#     output_file = open(log_file_name, 'a')
#     output_file.write(output_data)
#     output_file.close()
#
#     if epoch_i % 1 == 0:
#         torch.save(model.state_dict(), "./%s/net_params_%d.pkl" % (model_dir, epoch_i))  # save only the parameters
