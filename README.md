# TSNet:A Two-stage  Network for Image Dehazing with Multi-scale Fusion and Adaptive Learning

Abstract: Image dehazing has been a popular topic of research for a long time. 
 Previous deep learning-based image dehazing methods have failed to achieve satisfactory dehazing effects on both synthetic datasets 
 and real-world datasets, exhibiting poor generalization. Moreover, single-stage networks often result in many regions with artifacts and 
 color distortion in output images. To address these issues, this paper proposes a two-stage image dehazing network called TSNet, mainly 
 consisting of the multi-scale fusion module (MSFM) and the adaptive learning module (ALM). Specifically, MSFM and ALM enhance the generalization 
 of TSNet. The MSFM can obtain large receptive fields at multiple scales and integrate features at different frequencies to reduce the differences 
 between inputs and learning objectives. The ALM can actively learn of regions of interest in images and restore texture details more effectively. 
 Additionally, TSNet is designed as a two-stage network, where the first-stage network performs image dehazing, and the second-stage network is employed 
 to improve issues such as artifacts and color distortion present in the results of the first-stage network. We also change the learning objective from ground 
 truth images to opposite fog maps, which improves the learning efficiency of TSNet. Extensive experiments demonstrate that TSNet exhibits superior dehazing performance 
on both synthetic and real-world datasets compared to previous state-of-the-art methods.

# Requirement:

Python 3.9

Pytorch 1.12

CUDA 11.3

...
# Framework:


 ![2](https://github.com/zzhlovexuexi/TSNet/assets/126560356/c4f6351c-5a0f-41cb-b20a-5380658c127f)
 ![3](https://github.com/zzhlovexuexi/TSNet/assets/126560356/e2d18948-5536-4a62-aafd-07dab6e88756)
 ![5](https://github.com/zzhlovexuexi/TSNet/assets/126560356/53af0334-fb64-4891-85c9-ca7bb696ff3b)
  
# About dataset:

Since my code refers to Dehazeformer, the dataset format is the same as that in Dehazeformer. In order to avoid errors when 
training the datasets, please download the datasets from Dehazeformer for training.(https://github.com/IDKiro/DehazeFormer)

# Help:

If you have any questions, you can send an email to zzh07@tju.edu.cn or gxl@tju.edu.cn

# Thanks

Special Thanks to my supervisor, she gaves me selfless help in completing this work and answered my questions. Thank you very much.

