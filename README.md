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

## 🛠️ Prepare

### 📂 Feature Acquisition

You can obtain the original feature files from the following link:

- **features**: [Download from FakingRecipe repository](https://github.com/ICTMCG/FakingRecipe)

---

### 📝 Pseudo Labels Preparation

You should insert the pseudo labels from the *pseudo_label* folder into the corresponding *meta.json* files based on the video IDs.

- *pseudo_label*: Folder containing the generated pseudo labels for each video.
- *metainfo.json*: Metadata file where you need to add the pseudo label information for each video.
