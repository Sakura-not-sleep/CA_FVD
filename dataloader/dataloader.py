import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class CA_FVD_Dataset(Dataset):
    def __init__(self, vid_path,dataset):
        self.dataset=dataset
        if dataset=='fakesv': 
            self.data_all = pd.read_json('./fea/fakesv/metainfo.json',orient='records',dtype=False,lines=True)
            self.vid=[]
            with open(vid_path, "r") as fr:
                for line in fr.readlines():
                    self.vid.append(line.strip())
            self.data = self.data_all[self.data_all.video_id.isin(self.vid)]
            self.data.reset_index(inplace=True)

            self.text_semantic_fea_path='./fea/fakesv/preprocess_text/sem_text_fea.pkl'
            with open(self.text_semantic_fea_path, 'rb') as f:
                self.text_semantic_fea = torch.load(f)

            self.text_emo_fea_path='./fea/fakesv/preprocess_text/emo_text_fea.pkl'
            with open(self.text_emo_fea_path, 'rb') as f:
                self.text_emo_fea = torch.load(f)

            self.audio_fea_path='./fea/fakesv/preprocess_audio'
            self.visual_fea_path='./fea/fakesv/preprocess_visual'
        elif dataset=='fakett':
            self.data_all=pd.read_json('./fea/fakett/metainfo.json',orient='records',lines=True,dtype={'video_id': str})
            self.vid=[]
            with open(vid_path, "r") as fr:
                for line in fr.readlines():
                    self.vid.append(line.strip())
            self.data = self.data_all[self.data_all.video_id.isin(self.vid)]
            self.data.reset_index(inplace=True)

            self.text_semantic_fea_path='./fea/fakett/preprocess_text/sem_text_fea.pkl'
            with open(self.text_semantic_fea_path, 'rb') as f:
                self.text_semantic_fea = torch.load(f)

            self.text_emo_fea_path='./fea/fakett/preprocess_text/emo_text_fea.pkl'
            with open(self.text_emo_fea_path, 'rb') as f:
                self.text_emo_fea = torch.load(f)

            self.audio_fea_path='./fea/fakett/preprocess_audio'
            self.visual_fea_path='./fea/fakett/preprocess_visual'

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid = item['video_id']
        label = 1 if item['annotation']=='fake' else 0
        va_label = 1 if item['va'] == '1' else 0
        vt_label = 1 if item['vt'] == '1' else 0
        at_label = 1 if item['at'] == '1' else 0
        label = torch.tensor(label)
        va_label = torch.tensor(va_label)
        vt_label = torch.tensor(vt_label)
        at_label = torch.tensor(at_label)

        all_phrase_semantic_fea=self.text_semantic_fea['last_hidden_state'][vid]
        all_phrase_emo_fea=self.text_emo_fea['pooler_output'][vid]

        a_fea_path=os.path.join(self.audio_fea_path,vid+'.pkl')
        raw_audio_emo=torch.load(open(a_fea_path,'rb'))

        v_fea_path=os.path.join(self.visual_fea_path,vid+'.pkl')
        raw_visual_frames=torch.tensor(torch.load(open(v_fea_path,'rb')))

        return {
            'vid': vid,
            'label': label,
            'at_label': at_label,
            'va_label': va_label,
            'vt_label': vt_label,
            'all_phrase_semantic_fea': all_phrase_semantic_fea,
            'all_phrase_emo_fea': all_phrase_emo_fea,
            'raw_visual_frames': raw_visual_frames,
            'raw_audio_emo': raw_audio_emo,
        }

def pad_frame_sequence(seq_len,lst):
    attention_masks = []
    result=[]
    for video in lst:
        video=torch.FloatTensor(video)
        ori_len=video.shape[0]
        if ori_len>=seq_len:
            gap=ori_len//seq_len
            video=video[::gap][:seq_len]
            mask = np.ones((seq_len))
        else:
            video=torch.cat((video,torch.zeros([seq_len-ori_len,video.shape[1]],dtype=torch.float)),dim=0)
            mask = np.append(np.ones(ori_len), np.zeros(seq_len-ori_len))
        result.append(video)
        mask = torch.IntTensor(mask)
        attention_masks.append(mask)
    return torch.stack(result), torch.stack(attention_masks)


def collate_fn_CA_FVD(batch):
    num_visual_frames=83

    vid = [item['vid'] for item in batch]
    at_label = torch.stack([item['at_label'] for item in batch])
    va_label = torch.stack([item['va_label'] for item in batch])
    vt_label = torch.stack([item['vt_label'] for item in batch])
    label = torch.stack([item['label'] for item in batch])
    all_phrase_semantic_fea = [item['all_phrase_semantic_fea'] for item in batch] 
    all_phrase_emo_fea = torch.stack([item['all_phrase_emo_fea'] for item in batch]) 

    raw_visual_frames = [item['raw_visual_frames'] for item in batch]
    raw_audio_emo = [item['raw_audio_emo'] for item in batch]

    content_visual_frames, _ = pad_frame_sequence(num_visual_frames,raw_visual_frames)
    raw_audio_emo = torch.cat(raw_audio_emo,dim=0) 

    all_phrase_semantic_fea=[x if x.shape[0]==512 else torch.cat((x,torch.zeros([512-x.shape[0],x.shape[1]],dtype=torch.float)),dim=0) for x in all_phrase_semantic_fea] #batch*512*768
    all_phrase_semantic_fea=torch.stack(all_phrase_semantic_fea)

    return {
        'vid': vid,
        'label': label,
        'at_label': at_label,
        'va_label': va_label,
        'vt_label': vt_label,
        'all_phrase_semantic_fea': all_phrase_semantic_fea,
        'all_phrase_emo_fea': all_phrase_emo_fea,
        'raw_visual_frames': content_visual_frames,
        'raw_audio_emo': raw_audio_emo,
    }
