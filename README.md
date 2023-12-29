# DDPM-Torch
Interesting Topic as a part of **Generative AI** using **Diffusion** to generate different Bitmoji Faces based on Gender condition.

## Description:
***Denoising Diffusion Probabilistic Model [DDPM]***, a foundation model for famous *Stable Diffusion*, generate new samples from random noise. DDPM works by iteratively adding noise to an input signal (image) and then learning to denoise from the noisy signal to generate new samples. 

In this a approach called, **Classifier-Free Guidance** is used to condition the Diffusion process to generate new bitmoji faces based on the gender condition. 

This Diffusion method is inspired from the [paper](https://arxiv.org/abs/2006.11239) published on 2020.

It comprises of two process: *Forward Process (or) Diffusion Process* and *Reverse Process*.

## Dataset: 
[Bitmoji Faces](https://www.kaggle.com/datasets/mostafamozafari/bitmoji-faces)

## Video:
Below video shows the sequence of images generated while Sampling through timesteps.

https://github.com/harishhirthi/DDPM-Torch/assets/43694283/44ab0f17-0d39-4505-93af-597757d3af71

## References:
1. [DDPM Paper](https://arxiv.org/abs/2006.11239)
2. [YouTube](https://youtu.be/vu6eKteJWew?si=FnFf2A5AyYNtNiqF)
3. [YouTube](https://youtu.be/TBCRlnwJtZU?si=ydKy2uFuvEutYdXV)
4. [Github](https://github.com/hkproj/pytorch-ddpm)

## Contains:
* Modular Python Files - Class module for Diffusion Process
* Jupyter Notebook - Explains about Training and Sampling
* Config.yaml - Parameters
* Folder.



