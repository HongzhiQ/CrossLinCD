# CrossLinCD
The data and code for the paper: Generalizable Cross-Lingual Cognitive Distortion Detection with  Standardized Annotations and Multi-Task Learning

It contains:

* **public cognitive datasets**: All publicly available dataset srelated  to cognitive distortions to evaluate generalization across datasets.
* **Standardized alignment dataset**: The first bilingual cognitive distortion dataset with a unified annotation standard and its corresponding cognitive reasoning chain.
* **Code for evaluating cross-lingual model generalization**: It includes code for single-task, multi-task, and teacher-student model training tasks, aimed at systematically evaluating and improving the generalization of existing cross-lingual cognitive distortion recognition models.
* **Training and Inference of Large Language Models**:  Resources for training and inference using the large language model.
  
## 1. Dataset
### 1.1 Public cognitive datasets

* **Download the datasets**: 
  * **SocialCD-3k [1]**: [https://github.com/HongzhiQ/SupervisedVsLLM-EfficacyEval](https://github.com/HongzhiQ/SupervisedVsLLM-EfficacyEval)
  * **C2D2 Dataset [2]:** [https://github.com/bcwangavailable/C2D2-Cognitive-Distortion](https://github.com/bcwangavailable/C2D2-Cognitive-Distortion)
  * **Cognitive Reframing Dataset [3]:**[(https://github.com/behavioral-data/Cognitive-Reframing)](https://github.com/behavioral-data/Cognitive-Reframing)
  * **Therapist Dataset [4]:** [https://www.kaggle.com/datasets/arnmaud/therapist-qa](https://www.kaggle.com/datasets/arnmaud/therapist-qa)

### 1.2 The proposed standardized alignment dataset
After re-labeling the existing public dataset, we propose a standardized alignment cognitive distortion dataset with 12 labels for multi-label classification, along with its corresponding cognitive reasoning chain.

## 2. Code
### 2.1 Deep learning training


## References
1. Qi H, Zhao Q, Song C, et al. Evaluating the efficacy of supervised learning vs large language models for identifying cognitive distortions and suicidal risks in chinese social media[J]. arXiv preprint arXiv:2309.03564, 2023.
2. Wang B, Deng P, Zhao Y, et al. C2D2 Dataset: A Resource for the Cognitive Distortion Analysis and Its Impact on Mental Health[C]//Findings of the Association for Computational Linguistics: EMNLP 2023. 2023: 10149-10160.
3. Sharma A, Rushton K, Lin I W, et al. Cognitive reframing of negative thoughts through human-language model interaction[J]. arXiv preprint arXiv:2305.02466, 2023.
4. Shreevastava S, Foltz P. Detecting cognitive distortions from patient-therapist interactions[C]//Proceedings of the Seventh Workshop on Computational Linguistics and Clinical Psychology: Improving Access. 2021: 151-158.
