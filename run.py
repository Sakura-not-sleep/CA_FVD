from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataloader.dataloader import *
from train.Trainer import *
from model.CA_FVD import *

class Run():
    def __init__(self,config):
        self.dataset = config['dataset']
        self.mode = config['mode']
        self.epoches = config['epoches']
        self.batch_size = config['batch_size']
        self.early_stop = config['early_stop']
        self.device = config['device']
        self.lr = config['lr']
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.gamma = config['gamma']
        self.path_ckp=config['path_ckp']
        self.path_tb=config['path_tb']

    def get_dataloader(self,data_path):
        dataset=CA_FVD_Dataset(data_path,self.dataset)
        collate_fn=collate_fn_CA_FVD
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
        return dataloader

    def main(self):
        self.model = CA_FVD(self.dataset)

        if self.dataset=='fakesv':
            data_split_dir='./data/FakeSV/data-split/'
            save_predict_result_path='./predict_result/FakeSV/'
        elif self.dataset=='fakett':
            data_split_dir='./data/FakeTT/data-split/'
            save_predict_result_path='./predict_result/FakeTT/'
            
        train_data_path=data_split_dir+'vid_time3_train.txt'
        test_data_path=data_split_dir+'vid_time3_test.txt'
        val_data_path=data_split_dir+'vid_time3_val.txt'

        data_load_time_start = time.time()
        train_dataloader=self.get_dataloader(train_data_path)
        test_dataloader=self.get_dataloader(test_data_path)
        val_dataloader=self.get_dataloader(val_data_path)
        dataloaders=dict(zip(['train','test','val'],[train_dataloader,test_dataloader,val_dataloader]))
        print ('data load time: %.2f' % (time.time() - data_load_time_start))
        trainer=Trainer(model=self.model,device=self.device,lr=self.lr,dataloaders=dataloaders,epoches=self.epoches,model_name='CA_FVD',save_predict_result_path=save_predict_result_path,alpha=self.alpha,beta=self.beta,gamma = self.gamma,early_stop=self.early_stop,save_param_path=self.path_ckp+self.dataset+"/",writer=SummaryWriter(self.path_tb+self.dataset+"/"))
        ckp_path=trainer.train()
        result=trainer.test(ckp_path)
