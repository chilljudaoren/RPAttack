# RPAttack

This source code accompanies the article `RPA: Recursive Perturbation-Based Universal Adversarial Attacks on Multimodal Generative Tasks', to appear in IEEE Transactions on Circuits and Systems for Video Technology (TCSVT) in 2025.

⚠️ **I’m in the final stretch before graduation and focused on my thesis defense and job hunting (or PhD applications), so I can’t maintain this repository for now. I’ll resume organizing and updating it after graduation—thanks for your understanding.**

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

## Contact

Please drop an e-mail to <qianyaguan@zust.edu> if you have any enquiry.
