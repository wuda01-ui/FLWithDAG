#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import argparse
import copy
import os
import pickle
import warnings
# import torchvision.models as models
import numpy as np
from tqdm import tqdm
from helpers.datasets import partition_data
from helpers.utils import KLDiv, setup_seed
from models.generator import Generator
import torch
import torch.nn.functional as F
from models.mhresnet8x import mhresnet18
from models.resnet8x import resnet18
from helpers.synthesizers_ours import AdvSynthesizer
from Recorder import Recorder

warnings.filterwarnings('ignore')
upsample = torch.nn.Upsample(mode='nearest', scale_factor=7)




def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=100,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=5,
                        help="number of users: K")

    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.5)')
    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar10', help="name \
                        of dataset")
    parser.add_argument('--iid', type=int, default=0,
                        help='Default set to IID. Set to 0 for non-IID.')

    # Data Free
    parser.add_argument('--adv', default=10, type=float, help='scaling factor for adv loss')

    parser.add_argument('--bn', default=0, type=float, help='scaling factor for BN regularization')
    parser.add_argument('--oh', default=1, type=float, help='scaling factor for one hot loss (cross entropy)')
    parser.add_argument('--act', default=1, type=float, help='scaling factor for activation loss used in DAFL')

    parser.add_argument('--partition', default='dirichlet', type=str)
    parser.add_argument('--beta', default=0.5, type=float,
                        help=' If beta is set to a smaller value, '
                             'then the partition is more unbalanced')

    # Basic
    parser.add_argument('--lr_g', default=1e-3, type=float,
                        help='initial learning rate for generation')
    parser.add_argument('--input_channels', default=3, type=int, help='')
    parser.add_argument('--T', default=20, type=float)
    parser.add_argument('--g_steps', default=30, type=int, metavar='N',
                        help='number of iterations for generation')
    parser.add_argument('--batch_size', default=256, type=int, metavar='N',
                        help='number of total iterations in each epoch')
    parser.add_argument('--nz', default=256, type=int, metavar='N',
                        help='number of total iterations in each epoch')
    parser.add_argument('--synthesis_batch_size', default=256, type=int)
    # Misc
    parser.add_argument('--seed', default=1, type=int,
                        help='seed for initializing training.')
    parser.add_argument('--device', default=0, type=int,
                        help='seed for initializing training.')
    parser.add_argument('--model', default="cnn", type=str,
                        help='seed for initializing training.')

    args = parser.parse_args()
    return args


# class Ensemble(torch.nn.Module):
#     def __init__(self, model_list):
#         super(Ensemble, self).__init__()
#         self.models = model_list

#     def forward(self, x):
#         logits_total = 0
#         logits_list=[]
#         feats=[]
#         for i in range(len(self.models)):
#             logits ,feat= self.models[i](x,out_feature=True)
#             logits_list.append(logits)
#             feats.append(feat)
#             logits_total += logits
#         logits_e = logits_total / len(self.models)

#         return logits_e,logits_list,feats
class Ensemble(torch.nn.Module):
    def __init__(self, model_list,w):
        super(Ensemble, self).__init__()
        self.models = model_list
        self.w = w

    def forward(self, x):
        logits_total = 0
        logits_list=[]
        feats=[]
        for i in range(len(self.models)):
            logits ,feat= self.models[i](x,out_feature=True)
            logits = w[i]*logits
                    

            logits_list.append(logits)
            feats.append(feat)
            logits_total += logits
        logits_e = logits_total 

        return logits_e,logits_list

def criterion_feat(t_f, s_f):
    delta = torch.abs(s_f - t_f)
    loss_feat = torch.mean(delta.sum(1))
    return loss_feat
def kd_train(synthesizer, model,  optimizer, device,w):
    student, teacher = model
    student.train()
    teacher.eval()
    description = "loss={:.4f} acc={:.2f}%,loss_ensem={:.4f},loss_feat={:.4f},loss_mh={:.4f}"
    total_loss = 0.0
    total_loss1=0.0
    total_loss2=0.0
    total_loss3=0.0
    correct = 0.0
    with tqdm(synthesizer.get_data()) as epochs:
        for idx, (images) in enumerate(epochs):
            optimizer.zero_grad()
            images = images.cuda(device)
            with torch.no_grad():
                t_out,t_outs ,t_feats= teacher(images)
            s_outs,s_feats= student(images.detach(),out_feature=True)

            # s_out=torch.stack(s_outs).mean(0)
            logits_total = 0

            for i in range(len(s_outs )):
                logits = w[i]*s_outs[i]
                
                logits_total += logits
            
            # s_out =torch.stack(s_outs).mean(0)
            s_out=logits_total

            loss_ensem =F.l1_loss(s_out, t_out.detach())
            ##0.001 wanz diuf diumh quandiu
            loss_feat = 0*torch.stack(
                        [criterion_feat(s_feat, t_feat.detach()) for s_feat, t_feat in zip(s_feats, t_feats)]).mean()
            loss_mh = torch.stack(
                    [F.l1_loss(s_logit, t_logit.detach()) for s_logit, t_logit in zip(s_outs, t_outs)]).mean()  
            
            loss_s=(loss_ensem+loss_feat+loss_mh)
            loss_s.backward()
            optimizer.step()

            total_loss += loss_s.detach().item()
            avg_loss = total_loss / (idx + 1)
            total_loss1 += loss_ensem.detach().item()
            avg_loss1 = total_loss1 / (idx + 1)
            total_loss2 += loss_feat.detach().item()
            avg_loss2 = total_loss2 / (idx + 1)
            total_loss3 += loss_mh.detach().item()
            avg_loss3 = total_loss3 / (idx + 1)
            pred = s_out.argmax(dim=1)
            target = t_out.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
            acc = correct / len(synthesizer.data_loader.dataset) * 100

            epochs.set_description(description.format(avg_loss, acc,avg_loss1,avg_loss2,avg_loss3))

def test(model, test_loader, device):
    device = device
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(device), target.cuda(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()  # sum up batch loss
            pred = torch.max(output, 1)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    print('\n Test_set: Average loss: {:.4f}, Accuracy: {:.4f}\n'
          .format(test_loss, acc))
    return acc, test_loss
def mhtest(model, test_loader, device,w):
    device = device
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(device), target.cuda(device)
            outputs = model(data)
            # output=torch.stack(outputs).mean(0)
            logits_total = 0

            for i in range(len(outputs )):
                logits = w[i]*outputs[i]
                
                logits_total += logits
            
            # s_out =torch.stack(s_outs).mean(0)
            output=logits_total

            test_loss += F.cross_entropy(output, target, size_average=False).item()  # sum up batch loss
            pred = torch.max(output, 1)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    print('\n Test_set: Average loss: {:.4f}, Accuracy: {:.4f}\n'
          .format(test_loss, acc))
    return acc, test_loss
def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    if is_best:
        torch.save(state, filename)




def get_whole_test_set(dataset_name):
    party_num = int(dataset_name.split("_")[-2])
    pkl_file = open(dataset_name, 'rb')
    data = pickle.load(pkl_file)
    features, targets = data[0]["test_X"], data[0]["test_y"]
    for i in range(1, party_num):
        feature = data[i]["test_X"]
        target = data[i]["test_y"]
        features = np.concatenate((features, feature), axis=0)
        targets = np.concatenate((targets, target))
    return features, targets

def convert_dataset(args, feature, target):
    transpose = (0, 3, 2, 1)
    if args.input_channels == 3:
        feature = np.transpose(feature, transpose)
    feature = torch.Tensor(feature)
    target = torch.Tensor(target)
    return feature, target

if __name__ == '__main__':

    args = args_parser()
    expe_name='{}-{}-{}'.format(args.dataset,  args.num_users,args.beta)
    recorder = Recorder(base_path='result/main/OURS',
                        exp_name=expe_name,
                        logger_name=__name__,
                        code_path=__file__)

    recorder.logger.info(args)
    save_dir=os.path.join(recorder.exp_path, 'image')
    # wandb.init(config=args,
    #            project="ont-shot FL")

    setup_seed(args.seed)
    # pdb.set_trace()
    if(args.dataset=='cifar100'):
        nclass=100
        
    else:
        nclass=10
    # BUILD MODEL

    local_model = resnet18(nclass).cuda(args.device)
    bst_acc = -1
    description = "inference acc={:.4f}% loss={:.2f}, best_acc = {:.2f}%"
    local_weights = []
    local_model.train()
    acc_list = []
    users = []
    
        # ===============================================
    
        # ===============================================

    train_dataset, test_dataset, user_groups, traindata_cls_counts = partition_data(
            args.dataset, args.partition, beta=args.beta, num_users=args.num_users)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256,
                                            shuffle=False, num_workers=4)


    local_weights = torch.load('{}_{}_{}.pkl'.format(args.dataset, args.num_users, args.beta))
  
    if(args.beta==0.3):
        w=[0.125 ,0.25  ,0.25 ,0.25, 0.125 ] 
    if(args.beta==0.5):
        w=[0.25 , 0.125 ,0.25 ,0.125 , 0.25]
    if(args.beta==1.0):
        w=[0.225, 0.1 ,0.225 ,0.225 ,0.225 ]
    

    model_list = []
    for i in range(len(local_weights)):
        net = copy.deepcopy(local_model)
        net.load_state_dict(local_weights[i])
        model_list.append(net)
    ensemble_model = Ensemble(model_list,w)

    # test(ensemble_model, test_loader, args.device)
    print("ensemble acc:")
    # test_accuracy = test_on_dataset(args, test_features, test_targets, ensemble_model, args.device, batch_size_test=args.batch_size, name="test", jd=True)

    # ===============================================
    global_model = mhresnet18(args.num_users,nclass).cuda(args.device)
    # ===============================================

    # data generator
    nz = args.nz
    nc = 3 
    img_size = 32 

    generator = Generator(nz=nz, ngf=64, img_size=img_size, nc=nc).cuda(args.device)
    args.cur_ep = 0

    
    synthesizer = AdvSynthesizer(ensemble_model, model_list, global_model, generator,
                                    nz=nz, num_classes=nclass, 
                                    iterations=args.g_steps, lr_g=args.lr_g,
                                    synthesis_batch_size=args.synthesis_batch_size,
                                    sample_batch_size=args.batch_size,
                                    adv=args.adv, bn=args.bn, oh=args.oh,
                                    save_dir=save_dir, dataset=args.dataset, device=args.device,test_loader=test_loader,w=w)
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    criterion = KLDiv(T=args.T)
    optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr,
                                momentum=0.9)
    global_model.train()
    distill_acc = []
    for epoch in tqdm(range(args.epochs)):
        # 1. Data synthesis
        synthesizer.gen_data(args.cur_ep, args.device)  # g_steps
        args.cur_ep += 1
        
        kd_train(synthesizer, [global_model, ensemble_model],  optimizer, args.device,w) 
            
            # # kd_steps
        # acc, test_loss = test(global_model, test_loader, args.device)
        acc, test_loss = mhtest(global_model, test_loader, args.device,w)
        distill_acc.append(acc)
        is_best = acc > bst_acc
        bst_acc = max(acc, bst_acc)
        _best_ckpt = 'df_ckpt/{}.pth'.format(expe_name)
        print("best acc:{}".format(bst_acc))
        save_checkpoint({
            'state_dict': global_model.state_dict(),
            'best_acc': float(bst_acc),
        }, is_best, _best_ckpt)
        print({'accuracy': acc})
        recorder.add_scalars_from_dict({
                                'accuracy': acc},
                                global_step=epoch)
        recorder.logger.info(f'Epoch: {epoch}, '
                f'acc: {acc}, '
            )

    print({"global_accuracy": distill_acc})
    np.save("distill_acc_{}.npy".format(args.dataset), np.array(distill_acc))

    # ===============================================


