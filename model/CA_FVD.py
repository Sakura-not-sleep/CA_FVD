from .attention import *
from .transformer_align import *

class model(torch.nn.Module):
    def __init__(self,dataset):
        super(model,self).__init__()
        if dataset=='fakesv':
            self.encoded_text_semantic_fea_dim=768
        elif dataset=='fakett':
            self.encoded_text_semantic_fea_dim=512
        self.input_visual_frames=83

        self.mlp_text_emo = nn.Sequential(nn.Linear(768,128),nn.ReLU(),nn.Dropout(0.1))
        self.mlp_text_semantic = nn.Sequential(nn.Linear(self.encoded_text_semantic_fea_dim,128),nn.ReLU(),nn.Dropout(0.1))
        self.mlp_vision = nn.Sequential(nn.Linear(512,128),nn.ReLU(),nn.Dropout(0.1))
        self.mlp_audio = nn.Sequential(torch.nn.Linear(768, 128), torch.nn.ReLU(),nn.Dropout(0.1))

        self.transformer_align_visual = TransformerAlign(embed_size=128, num_heads=2, num_layers=1)
        self.transformer_align_text = TransformerAlign(embed_size=128, num_heads=2, num_layers=1)

        self.co_attention_at = co_attention(d_k=128, d_v=128, n_heads=4, dropout=0.1, d_model=128,visual_len=1, sen_len=512, fea_v=128, fea_s=128, pos=False)
        self.co_attention_vt = co_attention(d_k=128, d_v=128, n_heads=4, dropout=0.1, d_model=128,visual_len=self.input_visual_frames, sen_len=512, fea_v=128, fea_s=128,pos=False)
        self.co_attention_va = co_attention(d_k=128, d_v=128, n_heads=4, dropout=0.1, d_model=128,visual_len=self.input_visual_frames, sen_len=1, fea_v=128, fea_s=128,pos=False)

        self.trm_emo = nn.TransformerEncoderLayer(d_model = 128, nhead = 2, batch_first = True)
        self.trm_at = nn.TransformerEncoderLayer(d_model = 128, nhead = 2, batch_first = True)
        self.trm_vt = nn.TransformerEncoderLayer(d_model=128, nhead=2, batch_first=True)
        self.trm_va = nn.TransformerEncoderLayer(d_model=128, nhead=2, batch_first=True)

        self.content_classifier_emo = nn.Sequential(nn.Linear(128, 64),nn.ReLU(),nn.Dropout(0.1),nn.Linear(64,2))
        self.content_classifier_at = nn.Sequential(nn.Linear(128, 64), nn.ReLU(),nn.Dropout(0.1), nn.Linear(64, 2))
        self.content_classifier_vt = nn.Sequential(nn.Linear(128, 64), nn.ReLU(),nn.Dropout(0.1), nn.Linear(64, 2))
        self.content_classifier_va = nn.Sequential(nn.Linear(128, 64), nn.ReLU(),nn.Dropout(0.1), nn.Linear(64, 2))

        self.content_classifier = nn.Sequential(nn.Linear(128 * 4, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 2))

    def forward(self,**kwargs):
        all_phrase_semantic_fea=kwargs['all_phrase_semantic_fea'] # torch.Size([128, 512, 768])
        all_phrase_emo_fea=kwargs['all_phrase_emo_fea']           # torch.Size([128, 768])
        raw_visual_frames=kwargs['raw_visual_frames']             # torch.Size([128, 83, 512])
        raw_audio_emo=kwargs['raw_audio_emo']                     # torch.Size([128, 768])

        f_audio=self.mlp_audio(raw_audio_emo).unsqueeze(1)             # 128 * 1 *128
        f_text=self.mlp_text_semantic(all_phrase_semantic_fea)         # 128 * 512 * 128
        f_text_emo=self.mlp_text_emo(all_phrase_emo_fea).unsqueeze(1)  # 128 * 1 *128
        f_visual = self.mlp_vision(raw_visual_frames)                  # 128 * 83 * 128

        f_visual = self.transformer_align_visual(f_visual)   # 128 * 83+1 * 128
        f_text = self.transformer_align_text(f_text)         # 128 * 512+1 * 128

        f_visual = f_visual[:, 1:, :]           # 128 * 83 *128
        fea_cls_visual = f_visual[:, 0:1, :]    # 128 * 1 * 128
        f_text = f_text[:, 1:, :]               # 128 * 512 *128
        fea_cls_text = f_text[:, 0:1, :]        # 128 * 1 * 128
        fea_cls_audio = f_audio                 # 128 * 1 * 128

        fusion_emo_fea = self.trm_emo(torch.cat((f_text_emo, f_audio), 1))  # 128 * 2 * 128
        fusion_emo_fea = torch.mean(fusion_emo_fea, 1)                                  # 128 * 1 *128

        content_a_1, content_t_1 = self.co_attention_at(v=f_audio, s=f_text, v_len=f_audio.shape[1],s_len=f_text.shape[1])
        content_a_1 = torch.mean(content_a_1, -2)
        content_t_1 = torch.mean(content_t_1, -2)
        fusion_fea_1 = self.trm_at(torch.cat((content_a_1.unsqueeze(1), content_t_1.unsqueeze(1)), 1))
        fusion_fea_1 = torch.mean(fusion_fea_1, 1)   # 128 * 1 * 128

        content_v_2, content_t_2 = self.co_attention_vt(v=f_visual, s=f_text, v_len=f_visual.shape[1],s_len=f_text.shape[1])
        content_v_2=torch.mean(content_v_2,-2)
        content_t_2=torch.mean(content_t_2,-2)
        fusion_fea_2=self.trm_vt(torch.cat((content_v_2.unsqueeze(1),content_t_2.unsqueeze(1)),1))
        fusion_fea_2=torch.mean(fusion_fea_2,1)      # 128 * 1 * 128

        content_v_3, content_a_3 = self.co_attention_va(v=f_visual, s=f_audio, v_len=f_visual.shape[1],s_len=f_audio.shape[1])
        content_v_3 = torch.mean(content_v_3, -2)
        content_a_3 = torch.mean(content_a_3, -2)
        fusion_fea_3 = self.trm_va(torch.cat((content_v_3.unsqueeze(1), content_a_3.unsqueeze(1)), 1))
        fusion_fea_3 = torch.mean(fusion_fea_3, 1)    # 128 * 1 * 128

        fea_emo = self.content_classifier_emo(fusion_emo_fea) # 128 * 1 * 2
        fea_at = self.content_classifier_at(fusion_fea_1)     # 128 * 1 * 2
        fea_vt = self.content_classifier_vt(fusion_fea_2)     # 128 * 1 * 2
        fea_va = self.content_classifier_va(fusion_fea_3)     # 128 * 1 * 2

        fus_fea_cat = torch.cat((fusion_emo_fea, fusion_fea_1, fusion_fea_2, fusion_fea_3), 1)
        fus_fea_cat = self.content_classifier(fus_fea_cat)

        return (fea_cls_visual,fea_cls_text,fea_cls_audio,
                fus_fea_cat, fea_emo, fea_at , fea_vt , fea_va)


class CA_FVD(torch.nn.Module):
    def __init__(self,dataset):
        super(CA_FVD,self).__init__()
        self.content=model(dataset=dataset)
        self.tanh = nn.Tanh()

    def forward(self,  **kwargs):
        (fea_cls_visual, fea_cls_text, fea_cls_audio,
         fus_fea_cat, fea_emo, fea_at, fea_vt, fea_va) = self.content(**kwargs)
        fus_fea_multiplication = fea_emo * fea_at * fea_vt * fea_va
        output = fus_fea_cat * self.tanh(fus_fea_multiplication)
        # output = fus_fea_multiplication * self.tanh(fus_fea_cat)
        return output,fus_fea_cat,fus_fea_multiplication,fea_cls_visual,fea_cls_text,fea_cls_audio