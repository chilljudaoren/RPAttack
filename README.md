# RPAttack

This source code accompanies the article `RPA: Recursive Perturbation-Based Universal Adversarial Attacks on Multimodal Generative Tasks', to appear in IEEE Transactions on Circuits and Systems for Video Technology (TCSVT) in 2025.

⚠️ **I’m in the final stretch before graduation and focused on my thesis defense and job hunting (or PhD applications), so I can’t maintain this repository for now. I’ll resume organizing and updating it after graduation—thanks for your understanding.**

## Abstract
Current adversarial attacks pose a serious threat to the robustness of visual-language models (VLMs), including vision-language pre-trained models (VLPMs) and multimodal large language models (MLLMs). Traditional adversarial attacks are example-specific and rely on specific datasets. This practice suffers from low transferability and additional computation cost, while universal adversarial perturbations (UAPs) offer example-agnostic solutions by generalizing across inputs. However, current UAP methods mainly target VLPMs, demonstrating limited transferability and effectiveness in MLLMs. To bridge this gap, we propose the \underline{R}ecursive \underline{P}erturbation \underline{A}ttack (\textbf{RPA}), a novel black-box UAP method for both VLPMs and MLLMs. RPA employs a recursive perturbations strategy, utilizing token filtering and polynomial sampling methods to generate perturbations, thereby achieving incremental disruption and enhancing the transferability of the attack. To further enhance the effectiveness of the attack, RPA integrates a three-tier modality decoupling strategy, disentangling intra-modal, cross-modal, and fusion-modal features to effectively disrupt feature alignment and interactions. Extensive experiments validate that RPA achieves superior attack performance compared to existing UAP approaches. This work highlights new security concerns in multimodal AI systems and provides insights into the design of more robust models.

## Setup
### Install dependencies
We provide the environment configuration file exported by Anaconda, which can help you build up conveniently.
```bash
conda env create -f environment.yml
conda activate RPA
```  
### Prepare datasets and models

- Download the datasets, [Flickr30K](https://shannon.cs.illinois.edu/DenotationGraph/), [MSCOCO](https://cocodataset.org/#home), and [Nocaps](https://nocaps.org/), and fill the `image_root` in the configuration files.

- Download the checkpoints of the finetuned VLPMs and MLLMs: [BLIP](https://github.com/salesforce/BLIP), [X-VLM](https://github.com/zengyan-97/X-VLM), [BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2), and [InstructBLIP](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip)

## Training
<!--Download](https://drive.google.com/drive/folders/1r1rFpiif4Juv-tlFSRtrBAuWjlm0CjNY?usp=sharing) the UAP of RPA.-->
After downloading the corresponding packages, datasets, and model weights, you can start training the RPA.
```bash
python train_uap.py
```

## Contact

Please drop an e-mail to <qianyaguan@zust.edu> if you have any enquiry.

Copyright © 2025 IEEE. Personal use of this material is permitted. However, permission to use this material for any other purposes must be obtained from the IEEE by sending an email to pubs-permissions@ieee.org
