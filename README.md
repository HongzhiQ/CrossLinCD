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
### overview
This repository includes code for single-task, multi-task, and teacher-student model training tasks, aimed at systematically evaluating and improving the generalization of existing cross-lingual cognitive distortion recognition models.
* **Single task learning**：Evaluated the generalization performance of the trained models by testing them on all datasets, including unseen ones.
* **Multi task learning**： The model is trained and evaluated on multiple datasets, leveraging a shared encoder with distinct classification heads for task-specific predictions. The multi-task learning approach helps improve generalization by exposing the model to diverse annotation schemes.
* **Teacher student training strategy**: Trains a teacher model on the C2D2 dataset, which generates soft labels for the student model. The student model was trained on both the target task data and the corresponding soft-label data pairs.
* **Large language model fine tuning**: 
* Finally, to improve thegeneralizationof cognitivedistortions across language models,we re-annotated three public datasets and provided a detailed annotation process, denoted as standardized alignment dataset. 

### 2.1 Deep learning training
* **Single task learning**:
   * **`Train_Single_task_learning.py`**:Trains the single task learning model.
   * **`Predict_Single_task_learning.py`**: Predicts using the trained single task model.
* **Multi task learning**:
   * **`Train_Multi_task_learning.py`**:Trains the multi task learning model.
   * **`Predict_Multi_task_learning.py`**: Predicts using the trained multi task model.
* **Teacher student training strategy**: 
   * **`Train_Teacher_model.py`**: Trains the teacher model on the C2D2 dataset.
   * **`Predict_Teacher_model_C2D2.py`**: Evaluates the teacher model and generates soft labels.
   * **`Predict_Single_task_learning_StudentArchitecture.py`**: Uses the trained teacher model to generate soft labels for publicly available multi-label datasets.
   * **`Train_Single_task_learning_StudentArchitecture.py`**: Trains the student model using both soft and hard labels.
   * **`Train_MultiTaskWithTS_Architecture.py`**: Trains a model that combines multi-task learning with the teacher-student strategy.
   * **`Predict_MultiTaskWithTS_Architecture.py`**: Predicts using the trained MT + TS model. 
* **Training and Evaluation on Standardized Alignment Dataset**:
   * **`Train-alignmentData.py`**:Trains the model using standardized alignment dataset.
   * **`Predict-alignmentData.py`**: Predicts using the trained model.
### 2.2 Large language model fine tuning
* The training and inference of LLM are based on the LLaMA Factory [5] framework ([link](https://github.com/hiyouga/LLaMA-Factory/tree/main)). We publish details of our training here:  For all the LLMs, we set the batch size to 8, the numberof epochs to 5, and the learning rate to le-5, andtested the model that performed best on the validation set. The LLaMA3-8B-Chinese model was fine-tuned on the standardized alignment dataset for a maximum text length of 1500 tokens.
* Evaluating LLMs on downstream tasks：
 ```python
cd code/LLM
python evaluate.py
 ```
### Requirement

| Mandatory  | Recommend  |
|------|------|
| python  | 3.10  |
| torch  | 2.4.0  |
| transformers  | 4.49.0  |
| datasets  | 3.2.0  |
| accelerate  | 1.2.1  |
| peft  | 0.12.0 |
| trl  | 0.9.6  |
| tqdm | 4.66.4  |
| pandas |  2.0.3 |
| scikit-learn | 1.3.2  |


## References
1. Qi H, Zhao Q, Song C, et al. Evaluating the efficacy of supervised learning vs large language models for identifying cognitive distortions and suicidal risks in chinese social media[J]. arXiv preprint arXiv:2309.03564, 2023.
2. Wang B, Deng P, Zhao Y, et al. C2D2 Dataset: A Resource for the Cognitive Distortion Analysis and Its Impact on Mental Health[C]//Findings of the Association for Computational Linguistics: EMNLP 2023. 2023: 10149-10160.
3. Sharma A, Rushton K, Lin I W, et al. Cognitive reframing of negative thoughts through human-language model interaction[J]. arXiv preprint arXiv:2305.02466, 2023.
4. Shreevastava S, Foltz P. Detecting cognitive distortions from patient-therapist interactions[C]//Proceedings of the Seventh Workshop on Computational Linguistics and Clinical Psychology: Improving Access. 2021: 151-158.
5. Zheng Y, Zhang R, Zhang J, YeYanhan Y, Luo Z, et al. LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models[C]//Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics 2024.
