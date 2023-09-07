import torch, os, engine, classify, utils, sys
import numpy as np 
import torch.nn as nn
from sklearn.model_selection import train_test_split

dataset_name = "celeba"
device = "cuda"
root_path = "./target_model"
log_path = os.path.join(root_path, "target_logs")
model_path = os.path.join(root_path, "target_ckp")
os.makedirs(model_path, exist_ok=True)
os.makedirs(log_path, exist_ok=True)




def main(args, model_name, trainloader, testloader):
    n_classes = args["dataset"]["n_classes"]
    mode = args["dataset"]["mode"]
    n_epochs = args[model_name]['epochs']


    if model_name == "VGG16":
        if mode == "reg": 
            net = classify.VGG16(n_classes).to(device)
            net = torch.nn.DataParallel(net).to(device)
            
            optimizer = torch.optim.SGD(params=net.parameters(),
							    lr=args[model_name]['lr'], 
            					momentum=args[model_name]['momentum'], 
            					weight_decay=args[model_name]['weight_decay'])
            criterion = nn.CrossEntropyLoss().cuda()
            
            flag = 0
            if(flag == 0):
                epoch = 0
                print("a")
                print("------------------------------------------------------------------------------------")
                print(torch.cuda.is_available())
                print("------------------------------------------------------------------------------------")

            if(flag==1):
                name = input()
                checkpoint = torch.load('./target_models/target_ckp/{}.tar'.format(name),map_location=device)
                #checkpoint = torch.load('./VGG16.tar',map_location=device)
                print(checkpoint.keys())
                net.load_state_dict(checkpoint['state_dict'])
                print(net)
                if 'optimizer' in checkpoint.keys():
                    optimizer.load_state_dict(checkpoint['optimizer'])
                if 'epoch' in checkpoint.keys():
                    epoch = checkpoint['epoch']
                else:
                    epoch = 0
            #best_ACC = checkpoint['best_cl_acc']
            #print(best_ACC)
                print("=========================================")
                print('test accuracy is {} '.format(engine.test(net,criterion,testloader)))
                print('train accuracy is {} '.format(engine.test(net,criterion,trainloader)))
                print("=========================================")
            
        elif mode == "vib":
            net = classify.VGG16_vib(n_classes)
	
    elif model_name == "FaceNet":
        net = classify.FaceNet(n_classes).to(device)
        net = torch.nn.DataParallel(net).to(device)
        epoch = 0
        optimizer = torch.optim.SGD(params=net.parameters(),
							    lr=args[model_name]['lr'], 
            					momentum=args[model_name]['momentum'], 
            					weight_decay=args[model_name]['weight_decay'])
        criterion = nn.CrossEntropyLoss()

    
    print("Start Training!")
	
    if mode == "reg":
        best_model, best_acc = engine.train_reg(args, net, criterion, optimizer, trainloader, testloader, n_epochs,epoch)
    elif mode == "vib":
        best_model, best_acc = engine.train_vib(args, net, criterion, optimizer, trainloader, testloader, n_epochs,epoch)
	
    test_acc = engine.test(best_model, criterion, testloader)
    torch.save({'state_dict':best_model.state_dict(),'optimizer': optimizer.state_dict(),'epoch': epoch,'best_cl_acc': test_acc}, os.path.join(model_path, "{}_{:.2f}_allclass.tar").format(model_name, best_acc))



if __name__ == '__main__':
    file = "./config/classify.json"
    args = utils.load_json(json_file=file)
    model_name = args['dataset']['model_name']

    log_file = "{}.txt".format(model_name)
    utils.Tee(os.path.join(log_path, log_file), 'w')

    os.environ["CUDA_VISIBLE_DEVICES"] = args['dataset']['gpus']
    print(log_file)
    print("---------------------Training [%s]---------------------" % model_name)
    utils.print_params(args["dataset"], args[model_name], dataset=args['dataset']['name'])

    train_file = args['dataset']['train_file_path']
    print("train file is : {}".format(train_file))
    test_file = args['dataset']['test_file_path']
    print("test file is {}".format(test_file))

    _, trainloader = utils.init_dataloader(args, train_file, mode="train")
    _, testloader = utils.init_dataloader(args, test_file, mode="test")


    main(args, model_name, trainloader, testloader)