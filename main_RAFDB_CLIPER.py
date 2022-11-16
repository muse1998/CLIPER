import math
import numpy as np
import torchvision.models
import torch.utils.data as data
from torchvision import transforms
import cv2
import itertools
import torch.nn.functional as F
import pandas as pd
import os ,torch
import torch.nn as nn
import image_utils
import argparse,random
import models
import os
import confusion_matrix
from torchsummary import summary
from torchstat import stat
import clip
import random
from PIL import Image, ImageOps
from clip.simple_tokenizer import SimpleTokenizer
torch.set_printoptions(precision=3,edgeitems=14,linewidth=350)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raf_path', type=str, default='./annotation/', help='Raf-DB dataset path.')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Pytorch checkpoint file path')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Pretrained weights')
    return parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
##visualization of the learned text tokens
def visualize_word(ctx,token_embedding,topk):
    tokenizer = SimpleTokenizer()
    ctx=ctx.float()
    distance = torch.cdist(ctx, token_embedding)
    print(f"Size of distance matrix: {distance.shape}")
    sorted_idxs = torch.argsort(distance, dim=1)
    sorted_idxs = sorted_idxs[:, :topk]

    for m, idxs in enumerate(sorted_idxs):
        words = [tokenizer.decoder[idx.item()] for idx in idxs]
        # dist = [f"{distance[m, idx].item():.4f}" for idx in idxs]
        print(f"{m + 1}: {words}")
def _convert_image_to_rgb(image):
    return image.convert("RGB")
class RafDataSet(data.Dataset):
    def __init__(self, raf_path, phase):

        self.phase = phase
        self.raf_path = raf_path
        self.normalize_tr = transforms.Compose([
            transforms.Resize(224,interpolation = transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            _convert_image_to_rgb,
            transforms.ColorJitter(brightness=0.4,hue=0.3),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(224, scale=(0.15, 1), ratio=(0.3, 3.3)),
            # transforms.RandomErasing(scale=(0.02, 0.2), p=0.5),
        ])
        self.normalize_te = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), ])
        NAME_COLUMN = 0
        LABEL_COLUMN = 1
        df = pd.read_csv(os.path.join(self.raf_path, 'list_patition_label.txt'), sep=' ', header=None)
        if phase == 'train':
            dataset = df[df[NAME_COLUMN].str.startswith('train')]
        else:
            dataset = df[df[NAME_COLUMN].str.startswith('test')]
        file_names = dataset.iloc[:, NAME_COLUMN].values
        self.label = dataset.iloc[:, LABEL_COLUMN].values - 1 # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral

        self.file_paths = []
        # use raf aligned images for training/testing
        for f in file_names:
            f = f.split(".")[0]
            f = f +"_aligned.jpg"
            ## './RAFDB/aligned' is the root of the RAF-DB dataset
            path = os.path.join('./RAFDB/aligned', f)
            self.file_paths.append(path)


        self.aug_func = [image_utils.flip_image,image_utils.add_gaussian_noise]


    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        img = Image.open(os.path.join(path))

        label = self.label[idx]
        if self.phase == 'train':##training data preprocessing
            image=self.normalize_tr(img)
        else:##test data preprocessing
            image = self.normalize_te(img)
        return image, label, idx,path


def initialize_weight_goog(m, n=''):
    if isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        fan_out = m.weight.size(0)  # fan-out
        fan_in = 0
        if 'routing_fn' in n:
            fan_in = m.weight.size(1)
        init_range = 1.0 / math.sqrt(fan_in + fan_out)
        m.weight.data.uniform_(-init_range, init_range)
        m.bias.data.zero_()



def run_training():
    ##load clip model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    net, _ = clip.load("ViT-B/16", device=device)


    class get_text(nn.Module):
        def __init__(self,n_class,n_descriptor,clip_model):
            super().__init__()
            ##length of the learnable text tokens
            self.k_des = 4
            self.n_descriptor = n_descriptor
            ##random initialize the METD
            self.descriptor = nn.Parameter(torch.randn(n_class,n_descriptor*self.k_des,clip_model.ln_final.weight.shape[0],dtype=clip_model.dtype,requires_grad=True)/1000)
            ##number of the expression classes
            self.n_class = n_class

        def forward(self,text):
            # METD
            for i in range(self.n_descriptor):
                text[:,5+i*(self.k_des):5+i*(self.k_des)+self.k_des,:] = self.descriptor[:, i*self.k_des:i*self.k_des+self.k_des]
            return text
    ##replace the 'X' in raw METD with the learned text tokens. '7' is the number of the superclasses, '5' is the number of the subclasses of each subclass
    text_generator = get_text(n_class=7*5,n_descriptor=1,clip_model=net).to(device)
    ##initial datasets
    ##'./annotation/' is the root of the annotations
    train_dataset = RafDataSet('./annotation/', phase = 'train')

    print('Train set size:', train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = 128,
                                               num_workers = 4,
                                               shuffle = True,
                                               pin_memory = True)


    val_dataset = RafDataSet('./annotation/', phase = 'test')
    print('Validation set size:', val_dataset.__len__())

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size = 128,
                                               num_workers = 4,
                                               shuffle = False,
                                               pin_memory = True)





    ##stage 1:learning the proposed METD from the aligned text-image embedding space
    optimizer_v = torch.optim.AdamW(itertools.chain(text_generator.parameters()),0.01, weight_decay=0,eps=1e-4)
    ##gamma =1 means no lr decay in stage 1
    scheduler_v = torch.optim.lr_scheduler.ExponentialLR(optimizer_v, gamma=1)

    criterion = torch.nn.CrossEntropyLoss()




    ##best accuracy initialization
    best_acc=0
    for i in range(1, 60):
        running_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        ##stage 2: finetuning the image encoder
        if i==3:
            optimizer_v = torch.optim.AdamW(itertools.chain(net.visual.parameters()),0.000005, weight_decay=1e-1,eps =1e-4)
            ##cosine lr decay
            scheduler_v = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_v, T_0=10, T_mult=1)

        net.train()
        text_generator.train()
        ##fine-grained loss
        criterion_fine = torch.nn.CrossEntropyLoss(reduction='none')
        ##training process
        for batch_i, (images, targets, indexes,_) in enumerate(train_loader):

            iter_cnt += 1
            B, C, H, W = images.shape
            images = images.to(device)
            targets = targets.to(device)
            ##initialize METD, 'X' represents the learned text tokens. '7' is the number of the expression superclasses, '5' is the number of the subclasses of each subclass
            text_des = torch.cat([clip.tokenize(f"an ID photo of X X X X.") for i in range(7*5)]).to(device)
            ##text_embeddings generated by the text encoder
            embedding = net.token_embedding(text_des).type(net.dtype)
            text = text_generator(embedding)
            text_features = net.encode_text(text, text_des).float()
            ##image_embeddings generated by the image encoder
            image_features = net.encode_image(images)[0].float()

            image_feature = image_features / image_features.norm(dim=1, keepdim=True)
            text_feature = text_features / text_features.norm(dim=1, keepdim=True)

            ##cosine similarity between the image and text embedding, 0.01 is the temperature parameter of CLIP
            outputs = (image_feature @ text_feature.t())/0.01


            outputs_max = torch.max(outputs.view(B,7,-1),dim=2)[0]
            ##count the number of the samples of each subclass
            weight_t = torch.zeros(7,5).to(device)
            weight_sample= torch.zeros(B).to(device)
            for ii in range(B):
                weight_t[targets[ii],torch.max(outputs.view(B,7,-1),dim=2)[1][ii,targets[ii]]]+=1
            weight_t=torch.where(weight_t==0,-1/(torch.ones_like(weight_t)*99),weight_t)
            ## weight_sample is the the modulating factor alpha to force the network learn more on the subclasses with less sample
            for ii in range(B):
                weight_sample[ii]=weight_t[targets[ii]].sum()*torch.softmax(1/weight_t[targets[ii]],dim=0)[torch.max(outputs.view(B,7,-1),dim=2)[1][ii,targets[ii]]]/weight_t[targets[ii],torch.max(outputs.view(B,7,-1),dim=2)[1][ii,targets[ii]]]

            outputs_min =torch.min(outputs.view(B,7,-1),dim=2)[0]
            ##margin logits
            outputs_margin = torch.max(outputs.view(B,7,-1),dim=2)[0]
            ##fine-grained logits
            outputs_fg =  torch.zeros(B, 7*5-4).to(device)
            outputs_mean = torch.mean(outputs.view(B, 7, -1), dim=2)
            for ii in range(B):
                outputs_margin[ii,targets[ii]]=outputs_min[ii,targets[ii]]
                outputs_pos = outputs_max[ii,targets[ii]].unsqueeze(0)
                if targets[ii]==0:
                    outputs_neg = (outputs.view(B,7,-1)[ii,targets[ii]+1:7]).view(-1)
                elif targets[ii]==6:
                    outputs_neg = (outputs.view(B, 7, -1)[ii, 0:targets[ii]]).view(-1)
                else:
                    outputs_neg = torch.cat(((outputs.view(B, 7, -1)[ii, 0:targets[ii]]).view(-1),(outputs.view(B, 7, -1)[ii, targets[ii] + 1:7]).view(-1)), dim=0)

                outputs_fg[ii]= torch.cat((outputs_pos,outputs_neg),dim=0)
            ##visualization of the learned text tokens every 20 batches
            if i<=2 and batch_i%20==0:
                visualize_word(ctx=text_generator.descriptor.view(-1,512),token_embedding=net.token_embedding.weight,topk=5)
            # fine-grained cross-entropy loss and margin loss
            loss = criterion(outputs_margin, targets)+(weight_sample*criterion_fine(outputs_fg, torch.zeros(B).type(torch.int64).to(device))).mean()

            optimizer_v.zero_grad()
            loss.backward()
            optimizer_v.step()

            running_loss += loss
            _, predicts = torch.max(outputs_mean, 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num

        scheduler_v.step()

        acc = correct_sum.float() / float(train_dataset.__len__())
        running_loss = running_loss/iter_cnt
        print('[Epoch %d] Training accuracy: %.4f. Loss: %.3f' % (i, acc, running_loss))
        ##test process
        with torch.no_grad():
            running_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            sample_cnt = 0
            pre_lab_all = []
            Y_test_all = []
            net.eval()
            text_generator.eval()
            for batch_i, (images, targets, _,path) in enumerate(val_loader):
                B, C, H, W = images.shape
                images = images.to(device)
                targets = targets.to(device)
                ##initialize METD
                text_des = torch.cat([clip.tokenize(f"an ID photo of X X X X.") for i in range(7*5)]).to(device)
                ##text embedding
                with torch.no_grad():
                    embedding = net.token_embedding(text_des).type(net.dtype)
                text = text_generator(embedding)
                text_features = net.encode_text(text, text_des).float()


                # image embedding
                image_features = net.encode_image(images)[0].float()
                ##
                image_feature = image_features / image_features.norm(dim=1, keepdim=True)
                text_feature = text_features / text_features.norm(dim=1, keepdim=True)
                # cosine similarity between the image and text embedding, 0.01 is the temperature parameter of CLIP
                outputs=(image_feature @ text_feature.t())/0.01

                outputs_max = torch.max(outputs.view(B, 7, -1), dim=2)[0]
                ##count the number of the samples of each subclass
                weight_t = torch.zeros(7, 5).to(device)
                weight_sample = torch.zeros(B).to(device)
                for ii in range(B):
                    weight_t[targets[ii], torch.max(outputs.view(B, 7, -1), dim=2)[1][ii, targets[ii]]] += 1
                weight_t = torch.where(weight_t == 0, -1 / (torch.ones_like(weight_t) * 99), weight_t)
                ## weight_sample is the the modulating factor alpha to force the network learn more on the subclasses with less sample
                for ii in range(B):
                    weight_sample[ii] = weight_t[targets[ii]].sum() * torch.softmax(1 / weight_t[targets[ii]], dim=0)[torch.max(outputs.view(B, 7, -1), dim=2)[1][ii, targets[ii]]] / weight_t[targets[ii], torch.max(outputs.view(B, 7, -1), dim=2)[1][ii, targets[ii]]]


                outputs_min = torch.min(outputs.view(B, 7, -1), dim=2)[0]
                ##margin logits
                outputs_margin = torch.max(outputs.view(B, 7, -1), dim=2)[0]
                ##fine-grained logits
                outputs_fg = torch.zeros(B, 7 * 5 - 4).to(device)
                outputs_mean = torch.mean(outputs.view(B, 7, -1), dim=2)
                for ii in range(B):
                    outputs_margin[ii, targets[ii]] = outputs_min[ii, targets[ii]]
                    outputs_pos = outputs_max[ii, targets[ii]].unsqueeze(0)
                    if targets[ii] == 0:
                        outputs_neg = (outputs.view(B, 7, -1)[ii, targets[ii] + 1:7]).view(-1)
                    elif targets[ii] == 6:
                        outputs_neg = (outputs.view(B, 7, -1)[ii, 0:targets[ii]]).view(-1)
                    else:
                        outputs_neg = torch.cat(((outputs.view(B, 7, -1)[ii, 0:targets[ii]]).view(-1),
                                                 (outputs.view(B, 7, -1)[ii, targets[ii] + 1:7]).view(-1)), dim=0)

                    outputs_fg[ii] = torch.cat((outputs_pos, outputs_neg), dim=0)

                # fine-grained cross-entropy loss and margin loss
                loss = criterion(outputs_margin, targets) + (weight_sample * criterion_fine(outputs_fg,torch.zeros(B).type( torch.int64).to(device))).mean()




                running_loss += loss
                iter_cnt+=1
                _, predicts = torch.max(outputs_mean, 1)
                correct_num  = torch.eq(predicts,targets)
                bingo_cnt += correct_num.sum().cpu()
                sample_cnt += outputs.size(0)
                ##



                ##confusion matrix
                pre_lab = torch.argmax(outputs_mean, dim=1)
                confusion_Y_test = targets
                pre_lab = pre_lab.squeeze().cpu().numpy().tolist()
                confusion_Y_test = confusion_Y_test.squeeze().cpu().numpy().tolist()
                pre_lab_all.extend(pre_lab)
                Y_test_all.extend(confusion_Y_test)





            confusion_matrix.plot_confusion_matrix_2(pre_lab_all, Y_test_all)
            running_loss = running_loss/iter_cnt
            acc = bingo_cnt.float()/float(sample_cnt)
            acc = np.around(acc.numpy(),4)
            if acc>best_acc:
                best_acc = acc
            print("[Epoch %d] Validation accuracy:%.4f. Loss:%.3f, Best:%.3f" % (i, acc, running_loss,best_acc))








if __name__ == "__main__":
    run_training()
