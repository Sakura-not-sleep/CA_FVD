# Consistency-aware Fake Videos Detection on Short Video Platforms


## 📦 Dataset Preparation

Due to copyright reasons, we are unable to provide the original datasets.  You can download them from the following links:

### FakeSV

- **Description**: A multimodal benchmark for fake news detection on short video platforms.
- **Access**: [ICTMCG/FakeSV](https://github.com/ICTMCG/FakeSV)  
  📄 *FakeSV: A Multimodal Benchmark with Rich Social Context for Fake News Detection on Short Video Platforms*, AAAI 2023.

### FakeTT

- **Description**: A dataset for fake news detection from the perspective of creative manipulation.
- **Access**: [ICTMCG/FakingRecipe](https://github.com/ICTMCG/FakingRecipe)  
  📄 *FakingRecipe: Detecting Fake News on Short Video Platforms from the Perspective of Creative Process*, ACM MM 2024.

---

## 🛠️ Other Preparation

### 📂 Feature Acquisition

You can obtain the original feature files from the following link:

- **features:** [Download from FakingRecipe repository](https://github.com/ICTMCG/FakingRecipe)


### 📝 Pseudo Labels Preparation

You should insert the pseudo labels from the **pseudo_label** folder into the corresponding **metainfo.json** files based on the video IDs.

- **pseudo_label:** Folder containing the generated pseudo labels for each video.
- **metainfo.json:** Metadata file where you need to add the pseudo label information for each video.


- ### 🗂️ Code Structure

The overall structure of this project is organized as follows:

```text
CA_FVD/

├── data/
│   ├── FakeSV/
│   │   └── data-split/             
│   ├── FakeTT/
│   │   └── data-split/            
├── fea/                            
│   ├── fakesv/
│   │   ├── preprocess_audio/        
│   │   ├── preprocess_text/         
│   │   ├── preprocess_visual/      
│   │   └── metainfo.json
│   ├── fakett/
│   │   ├── preprocess_audio/
│   │   ├── preprocess_text/
│   │   ├── preprocess_visual/
│   │   └── metainfo.json
├── dataloader/
│   └── dataloader.py             
├── model/
│   ├── attention.py             
│   ├── CA_FVD.py                   
│   ├── transformer_align.py        
│   └── trm.py                      
├── pseudo_label/            
├── train/
│   ├── metrics.py                  
│   ├── Trainer.py               
├── main.py                     
├── requirements.txt               
├── run.py                          
```

---

## ⚙️ Environment Setup and Run

We used the following versions for development:  
- **Anaconda**: `24.5.0`  
- **Python**: `3.10.16`  
- **PyTorch**: `2.0.1`  
- **CUDA**: `11.7`

You can set up the environment by running the following commands:

```bash
# 1. Create a new conda environment named CA_FVD
conda create -n CA_FVD python=3.10.16

# 2. Activate the environment
conda activate CA_FVD

# 3. Install PyTorch 2.0.1 with CUDA 11.7 support
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# 4. Install the remaining dependencies
pip install -r requirements.txt --no-deps

# 5. run
python main.py

