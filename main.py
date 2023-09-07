import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import copy
import engine
import utils
import time
from generator import Generator
from classify import *
from utils import * 
from SAC import Agent
from attack import inversion

dataset_name = "celeba"
device = "cuda"
root_path = "./target_model"
log_path = os.path.join(root_path, "target_logs")
model_path = os.path.join(root_path, "target_ckp")
os.makedirs(model_path, exist_ok=True)
os.makedirs(log_path, exist_ok=True)

parser = argparse.ArgumentParser(description="RLB-MI")
parser.add_argument('-model_name', default='VGG16')
parser.add_argument("-max_episodes", type=int, default=40000)
parser.add_argument("-max_step", type=int, default=1)
parser.add_argument("-seed", type=int, default=42)
parser.add_argument("-alpha", type=float, default=0)
parser.add_argument("-n_classes", type=int, default=2)
parser.add_argument("-z_dim", type=int, default=100)
parser.add_argument("-n_target", type=int, default=100)

args = parser.parse_args()

print("Provide your target model ! ")
target_model_name = input()
print("Provide your generator model ! ")
generator_model_name = input()
print( "Provide your Evaluation model ! ")
evaluation_model_name = input()

def seed_everything(seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)  # type: ignore
        torch.backends.cudnn.deterministic = True  # type: ignore
        torch.backends.cudnn.benchmark = True  # type: ignore

if __name__ == "__main__":
    model_name = args.model_name
    max_episodes = args.max_episodes
    max_step = args.max_step
    seed = args.seed
    alpha = args.alpha
    n_classes = args.n_classes
    z_dim = args.z_dim
    n_target = args.n_target

    
    file = "./config/classify.json"
    argss = utils.load_json(json_file=file)

    print("Target Model : " + model_name)

    # Generator
    G = Generator(z_dim)
    G = nn.DataParallel(G).cuda()
    G = G.cuda()
    ckp_G = torch.load('weights/{}.tar'.format( generator_model_name ))['state_dict']
    #ckp_G = torch.load('weights/improved_celeba_G.tar')['state_dict']
    load_my_state_dict(G, ckp_G)
    G.eval()

    # Target model

    if model_name == "VGG16":
        T = VGG16(n_classes)
        path_T = './weights/{}.tar'.format( target_model_name )
    elif model_name == 'ResNet-152':
        T = IR152(n_classes)
        path_T = './weights/ResNet-152.tar'
    elif model_name == "Face.evoLVe":
        T = FaceNet64(n_classes)
        path_T = './weights/Face.evoLVe.tar'

    T = torch.nn.DataParallel(T).cuda()
    ckp_T = torch.load(path_T)
    T.load_state_dict(ckp_T['state_dict'], strict=False)
    T.eval()

    criterion = nn.CrossEntropyLoss()
    test_file = "./data/testset.txt"
    train_file = "./data/trainset.txt"
    #print(utils.init_dataloader(argss, test_file, mode="test"))
    _, testloader = utils.init_dataloader(argss, test_file, mode="test")
    _, trainloader = utils.init_dataloader(argss, train_file, mode="train")
    print('test accuracy is {} '.format(engine.test(T,criterion,testloader)))
    print('train accuracy is {} '.format(engine.test(T,criterion,trainloader)))

    print("============================================")
    print("aaa")
    # Evaluation model
    E = FaceNet(n_classes)
    path_E = './weights/{}.tar'.format( evaluation_model_name )

    E = torch.nn.DataParallel(E).cuda()
    ckp_E = torch.load(path_E)
    E.load_state_dict(ckp_E['state_dict'], strict=False)
    E.eval()

    # Method
    seed_everything(seed)
    total = 0
    cnt = 0
    cnt5 = 0
    identities = range(n_classes)
    print("identities is : {}".format(identities))
    print("n_target is : {}".format(n_target))
    targets = random.sample(identities, n_target)
    print("targets is : {}".format(targets))

    #total = 0
    #cnt = 0

    result_path = './target_models/attack_logs/{}.txt'.format( target_model_name )
    f = open( result_path, 'w' )
    

    for label_j in targets:
        agent = Agent(state_size=z_dim, action_size=z_dim, random_seed=seed, hidden_size=256, action_prior="uniform")
        recon_image = inversion(agent, G, T, alpha, z_dim=z_dim, max_episodes=max_episodes, max_step=max_step, label=label_j, model_name=model_name)
        _, output= E(low2high(recon_image))
        eval_prob = F.softmax(output[0], dim=-1)
        #print("eval_prob is : {}".format(eval_prob))
        top_idx = torch.argmax(eval_prob)
        _,top5_idx = torch.topk(eval_prob,5)

        total += 1
        if top_idx == label_j:
            cnt += 1

        if label_j in top5_idx:
            cnt5 += 1

        print("cnt is : {}".format(cnt))
        #print("cnt5 is : {}".format(cnt5))
        print("total is : {}".format(total))
        print(cnt5)
        acc = cnt / total
        #acc5 = cnt5 / total
        print("Classes {}/{}, Accuracy : {:.3f}".format(total, n_target, acc))
        #print("Classes {}/{}, Accuracy : {:.3f}, Top-5 Accuracy : {:.3f}".format(total, n_target, acc, acc5))

        # write file
        f.writelines( "Classes is : {}".format(label_j))
        f.write('\n')
        f.writelines( "Predict Classes is : {}".format(top_idx))
        f.write('\n')
        f.writelines( "cnt is : {}".format(cnt) )
        f.write('\n')
        #f.writelines( "cnt5 is : {}".format(cnt5) )
        #f.write('\n')
        f.writelines( "total is : {}".format(total) )
        f.write('\n')
    f.close()
        
