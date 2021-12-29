<div align="center">
  <h1>Human Mask Preprocessing Approach Using SilNet for Pose Generation Using Global Flow Local Attention</h1>
</div>

<p align="center">
  Project for 5561 Computer Vision at University of Minnesota
</p>


[Link to our paper](https://docs.google.com/document/d/1Txw2NzXJOc37jaZWqmg0giqN3Pi_TDfe3SaDmSIHFKY/edit?usp=sharing)


## Background
 Human pose image generation from keypoints is a task well suited for a Generative Adversarial Network (GAN), evidenced by current state-of-the-art approaches. This task is challenging, however, when the images feature a complex background. Many implementations train using entire photos, with noise introduced to the network due to the presence of a background. Additionally, the GAN is overloaded to predict both the person’s shape and the texture mapping simultaneously. Due to these limitations, we propose to 1) mask the input human from the background, 2) learn a predicted silhouette of the target pose, and 3) augment a baseline GAN, which learns the feature/texture mapping, with the predicted pose mask. We ultimately propose to train the models in 2) and 3) separately in order to break apart the task. These models will take architectural design from SilNet and Global Flow Local Attention (GFLA). This approach will allow one model to be specialised in the remapping of the human’s body to a new masked pose, and one specialised on the features and textures on the body. Our initial input mask eliminates the noise in the system caused by the background, while the separated model training approach simplifies the generator’s task of feature and texture warping by feeding it a predicted target silhouette. Our evaluation results demonstrate improved performance, as a result of more accurate shape prediction, as well as reduced artificating and blurring of the foreground human textures by removing the need to additionally generate a background. 

## Dataset
"
HUMBI is a large multiview image dataset of human body expressions (gaze, face, hand, body, and garment) with natural clothing. 107 synchronized HD cameras are used to capture more than 700 subjects across gender, ethnicity, age, and style. With the multiview image streams, it provides 3D mesh models. HUMBI is highly effective in learning and reconstructing a complete human model and is complementary to the existing datasets of human body expressions with limited views and subjects.
"
[HUMBI](https://humbi-data.net/)


## Models

### Global Flow Local Attention
Original Paper: [Deep Image Spatial Transformation for Person Image Generation](https://arxiv.org/abs/2003.00696)

### SilNet
Original Paper: [SilNet : Single- and Multi-View Reconstruction by Learning from Silhouettes](https://arxiv.org/abs/1711.07888)

Design Inspiration: [Pose-Guided Human Animation from a Single Image in the Wild](https://arxiv.org/abs/2012.03796)


## Contributing

- Anton King, king1266@umn.edu
- Carl Molnar, molna018@umn.edu
- Pranav Julakanti, julak004@umn.edu
- Xianjian Xie, xie00250@umn.edu


## License

[MIT 2021](LICENSE)
