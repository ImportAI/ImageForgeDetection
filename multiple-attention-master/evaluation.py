# from datasets.data import FF_dataset,Celeb_test,deeperforensics_dataset,dfdc_dataset
# from datasets.dataset import DeepfakeDataset
# from models.MAT import MAT
import pickle
import json
import time

import torch
import re
import os
from sklearn.metrics import roc_auc_score as AUC
import numpy as np
from copy import deepcopy
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

from MAT import netrunc
import pandas as pd

def load_model(name):
    with open('runs/%s/config.pkl'%name,'rb') as f:
        config=pickle.load(f)
    # net= MAT(**config.net_config)

    net = netrunc(config.net,config.feature_layer,config.num_classes,config.dropout_rate,config.pretrained)
    return config,net

#
def find_best_ckpt(name,last=False):
    if last:
        return len(os.listdir('checkpoints/%s'%name))-1
    with open('runs/%s/train.log'%name) as f:
        lines=f.readlines()[2::2]
    accs=[float(re.search('acc\\:(.*)\\,',a).groups()[0]) for a in lines]
    best=accs.index(max(accs))
    return best

def acc_eval(labels,preds):
    labels=np.array(labels)
    preds=np.array(preds)
    thres=0.5
    acc=np.mean((preds>=thres)==labels)
    return thres,acc


def test_eval(net,setting,testset):
    test_dataset=DeepfakeDataset(phase='test',**setting)
    test_loader=torch.utils.data.DataLoader(test_dataset, batch_size=setting['imgs_per_video'],shuffle=False,pin_memory=True,num_workers=8)
    for i, (X, y) in enumerate(test_loader):
        testset[i].append([])
        if -1 in y:
            testset[i].append(0.5)
            continue
        X = X.to('cuda',non_blocking=True)
        with torch.no_grad():
            for x in torch.split(X,20):
                logits=net(x)
                pred=torch.nn.functional.softmax(logits,dim=1)[:,1]
                testset[i][-1]+=pred.cpu().numpy().tolist()
        testset[i].append(np.mean(testset[i][-1]))


def test_metric(testset):
        frame_labels=[]
        frame_preds=[]
        video_labels=[]
        video_preds=[]
        for i in testset:
            frame_preds+=i[2]
            frame_labels+=[i[1]]*len(i[2])
            video_preds.append(i[3])
            video_labels.append(i[1])
        video_thres,video_acc=acc_eval(video_labels,video_preds)
        frame_thres,frame_acc=acc_eval(frame_labels,frame_preds)
        video_auc=AUC(video_labels,video_preds)
        frame_auc=AUC(frame_labels,frame_preds)
        rs={'video_acc':video_acc,'video_threshold':video_thres,'video_auc':video_auc,'frame_acc':frame_acc,'frame_threshold':frame_thres,'frame_auc':frame_auc}
        return rs

def dfdc_metric(testset):
    rs=test_metric(testset)
    video_preds=[]
    video_labels=[]
    for i in testset:
        video_preds.append(i[3])
        video_labels.append(i[1])
    video_preds=torch.tensor(video_preds).cuda()
    video_labels=torch.tensor(video_labels).cuda()
    video_preds=torch.stack([1-video_preds,video_preds],dim=1)
    rs['logloss']=torch.nn.functional.cross_entropy(video_preds,video_labels).item()
    return rs

def ff_metrics(testset):
    result=dict()
    temp_set=dict()
    for k,j in enumerate(['Origin','Deepfakes','NeuralTextures','FaceSwap','Face2Face']):
        d=testset[k*140:(k+1)*140]
        temp_set[j]=d

    for i in ['Deepfakes','NeuralTextures','FaceSwap','Face2Face','all']:
        if i!='all':
            rs=test_metric(temp_set[i]+temp_set['Origin'])
        else:
            rs=test_metric(testset) 
        result[i]=rs
    return result

def all_eval(name,ckpt=None,test_sets=['ff-all','celeb','deeper']):
    config,net=load_model(name)
    setting=config.val_dataset
    codec=setting['datalabel'].split('-')[2]
    setting['min_frames']=100
    setting['frame_interval']=5
    setting['imgs_per_video']=20
    setting['datalabel']='ff-all-%s'%codec
    list_of_files = os.listdir('checkpoints/%s'%name)
    list_of_files=list(map(lambda x:int(x[5:-4]),list_of_files))
    if ckpt is None:
        ckpt=find_best_ckpt(name)
    if ckpt<0:
        ckpt=max(list_of_files)+1+ckpt
    
    state_dict=torch.load('checkpoints/%s/ckpt_%s.pth'%(name,ckpt))['state_dict']
    net.load_state_dict(state_dict,strict=False)
    os.makedirs('evaluations/%s'%name,exist_ok=True)
    net.eval()
    net.cuda()
    result=dict()
    if 'ff-all' in test_sets:
        testset=[]
        for i in ['Origin','Deepfakes','NeuralTextures','FaceSwap','Face2Face']:
            testset+=FF_dataset(i,codec,'test')
        test_eval(net,setting,testset)
        with open('evaluations/%s/ff-test-%s.json'%(name,ckpt),'w') as f:
            json.dump(testset,f)
        result['ff']=ff_metrics(testset)
    if 'deeper' in test_sets:
        setting['datalabel']='deeper-'+codec
        testset=deeperforensics_dataset('test')+FF_dataset('Origin',codec,'test')
        test_eval(net,setting,testset)
        with open('evaluations/%s/deeper-test-%s.json'%(name,ckpt),'w') as f:
            json.dump(testset,f)
        result['deeper']=test_metric(testset)
    if 'celeb' in test_sets:
        setting['datalabel']='celeb'
        setting['min_frames']=100
        setting['frame_interval']=5
        setting['imgs_per_video']=20
        testset=deepcopy(Celeb_test)
        test_eval(net,setting,testset)
        with open('evaluations/%s/celeb-test-%s.json'%(name,ckpt),'w') as f:
            json.dump(testset,f)
        result['celeb']=test_metric(testset)
    if 'dfdc' in test_sets:
        setting['datalabel']='dfdc'
        setting['min_frames']=100
        setting['frame_interval']=5
        setting['imgs_per_video']=20
        testset=dfdc_dataset('test')
        test_eval(net,setting,testset)
        with open('evaluations/%s/dfdc-test-%s.json'%(name,ckpt),'w') as f:
            json.dump(testset,f)
        result['dfdc']=dfdc_metric(testset)
    with open('evaluations/%s/metrics-%s.json'%(name,ckpt),'w') as f:
        json.dump(result,f)

def eval_meancorr(name,ckpt=None):
    config,net=load_model(name)
    setting=config.val_dataset
    codec=setting['datalabel'].split('-')[2]
    setting['frame_interval']=5
    setting['imgs_per_video']=60
    setting['datalabel']='ff-all-%s'%codec
    if ckpt is None:
        ckpt=find_best_ckpt(name)
    if ckpt<0:
        ckpt=len(os.listdir('checkpoints/%s'%name))+ckpt
    state_dict=torch.load('checkpoints/%s/ckpt_%s.pth'%(name,ckpt))['state_dict']
    net.load_state_dict(state_dict,strict=False)
    net.eval()
    net.cuda()
    testset=[]
    test_dataset=DeepfakeDataset(phase='test',**setting)
    test_loader=torch.utils.data.DataLoader(test_dataset, batch_size=30,shuffle=False,pin_memory=True,num_workers=8)
    count=0
    mc_count=0
    for i, (X, y) in enumerate(test_loader):
        x = X.to('cuda',non_blocking=True)
        with torch.no_grad():
            count+=x.shape[0]
            layers = net.net(x)
            raw_attentions = layers[config.attention_layer]
            attention_maps=net.attentions(raw_attentions).flatten(-2)
            #print(attention_maps.shape)
            srs=torch.norm(attention_maps,dim=2)
            #print(srs.shape)
            for a in range(0,config.num_attentions-1):
                for b in range(a+1,config.num_attentions):
                    mc_count+=torch.sum(torch.sum(attention_maps[:,a,:]*attention_maps[:,b,:],dim=-1)/(srs[:,a]*srs[:,b]))
    return mc_count/(config.num_attentions-1)/config.num_attentions*2/count

                


def merge(g):
    if type(g[0])==float:
        return np.mean(g)
    else:
        c=dict()
        for i in g[0].keys():
            c[i]=merge([u[i] for u in g])
    return c


def gather_metrics(name,fl=None):
    path='evaluations/%s/'%name
    l=os.listdir(path)
    l=[path+i for i in l if i.startswith('metrics-')]
    if fl:
        l=list(filter(fl,l))
    g=[]
    for i in l:
        with open(i) as f:
            g.append(json.load(f))
    return merge(g)


def cal_auc(y,score):
    pass

def dump_result(data,file_path,column_names):
    """
    :param data: result dict
    :param file_path: output file path
    :param column_names: header names list
    :return:
    """
    with open(file_path,'a+') as f:
        if not f.readlines():
            lines = ' '.join(column_names)
            f.write(lines+'\n')


# 验证集推理，并将推理结果写入文件
def eval_result(net,test_loader,val_result_path):
    result_dict = {'img_name': [], 'label': [], 'preds': []}
    def write_data(path,data):
        with open(path, 'a') as f:
            for img_name, label, preds in zip(data['img_name'], data['label'],
                                              data['preds']):
                f.write(img_name + ' ' + str(label) + ' ' + str(preds) + '\n')

    with open(val_result_path,'w') as f:
        f.write('img_name'+' '+'label'+' '+'preds\n')
    for i, (X, y, name) in enumerate(test_loader):
        if i%10==0:
            # df_result = pd.DataFrame(result_dict)
            # print(df_result)
            # df_result.to_csv(val_result_path,mode='a',index=False)
            write_data(val_result_path,result_dict)
            result_dict={'img_name':[],'label':[],'preds':[]}
        y = y.numpy().tolist()
        result_dict['label']+=y
        result_dict['img_name']+=list(name)
        # if -1 in y:
        #     result_set[i].append(0.5)
        #     continue
        X = X.to('cuda',non_blocking=True)
        with torch.no_grad():
            logits=net(X)
            pred=torch.nn.functional.softmax(logits,dim=1)[:,1]
            result_dict['preds']+=pred.cpu().numpy().tolist()
    if result_dict:
        # df_result = pd.DataFrame(result_dict)
        # df_result.to_csv(val_result_path, mode='a', index=False)
        write_data(val_result_path, result_dict)

def evaluate_model(name,val_loader):
    time_mark = time.strftime("%Y-%m-%d",time.localtime())
    config, net = load_model(name)
    ckpt = find_best_ckpt(name)

    state_dict = torch.load('checkpoints/%s/ckpt_%s.pth' % (name, ckpt))['state_dict']
    net.load_state_dict(state_dict, strict=False)
    os.makedirs('evaluations/%s' % name, exist_ok=True)
    net.eval()
    net.cuda()

    eval_result_path = 'evaluations/%s/val_result_%s.txt' % (name,time_mark)
    eval_result(net,val_loader,eval_result_path)

    # TODO 修改计算代码
    df_result = pd.read_csv(eval_result_path,sep=' ')
    labels = df_result['label'].to_list()
    preds = df_result['preds'].to_list()

    # 计算auc
    fpr, tpr, threshold = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)

    return {"fpr": fpr, "tpr": tpr, "roc_auc": roc_auc}



if __name__=="__main__":
    from datasets.my_dataset import ForgeDataset
    data_root_path = os.path.abspath('../data')
    val_dataset = ForgeDataset(data_root_path, usage='val')
    test_loader = torch.utils.data.DataLoader(val_dataset,batch_size=5,
                                              collate_fn=val_dataset.collate_fn)
    for name in os.listdir('checkpoints'):
        print(name)
        if not name.endswith('b2'):
            continue
        # try:
            # all_eval(name)
        result = evaluate_model(name,test_loader)
        print(result)
        # except:
        #     pass