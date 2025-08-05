# Lung Cancer Screening Classification by Sequential Multi-Instance Learning (SMILE) Framework with Multiple CT Scans
This is the official implementation repository for the paper titled "Lung Cancer Screening Classification by Sequential Multi-Instance Learning (SMILE) Framework with Multiple CT Scans", which is published on IEEE Transactions on Medical Imaging. 

## Author List
Wangyuan Zhao, Yuanyuan Fu, Yujia Shen, Jingchen Ma, Lu Zhao, Xiaolong Fu, Puming Zhang and Jun Zhao.

## Abstract
Lung cancer screening with computed tomography (CT) scans can effectively improve the survival rate through the early detection of lung cancer, which typically identified in the form of pulmonary nodules. Multiple sequential CT images are helpful to determine nodule malignancy and play a significant role to detect lung cancers. It is crucial to develop effective lung cancer classification algorithms to achieve accurate results from multiple images without nodule location annotations, which can free radiologists from the burden of labeling nodule locations before predicting malignancy. In this study, we proposed the sequential multi-instance learning (SMILE) framework to predict high-risk lung cancer patients with multiple CT scans. SMILE included two steps. The first step was nodule instance generation. We employed the nodule detection algorithm with image category transformation to identify nodule instance locations within the entire lung images. The second step was nodule malignancy prediction. Models were supervised by patient-level annotations, without the exact locations of nodules. We embedded multi-instance learning with temporal feature extraction into a fusion framework, which effectively promoted the classification performance. SMILE was evaluated by five-fold cross-validation on a 925-patient dataset (182 malignant, 743 benign). Every patient had three CT scans, of which the interval period was about one year. Experimental results showed the potential of SMILE to free radiologists from labeling nodule locations. The source code will be available at https://github.com/wyzhao27/SMILE.

## Requirements
* Keras >= 3.0
* TensorFlow >= 2.16.0
* SimpleITK

## Citing the work
```
@ARTICLE{10960342,
  author={Zhao, Wangyuan and Fu, Yuanyuan and Shen, Yujia and Ma, Jingchen and Zhao, Lu and Fu, Xiaolong and Zhang, Puming and Zhao, Jun},
  journal={IEEE Transactions on Medical Imaging}, 
  title={Lung Cancer Screening Classification by Sequential Multi-Instance Learning (SMILE) Framework With Multiple CT Scans}, 
  year={2025},
  volume={44},
  number={8},
  pages={3151-3161},
  doi={10.1109/TMI.2025.3559143}}

```
