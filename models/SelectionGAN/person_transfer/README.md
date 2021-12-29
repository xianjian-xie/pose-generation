# Person Image Generation
Code for person image generation. This is Pytorch implementation for pose transfer on both Market1501 and DeepFashion dataset.

## Requirement
* pytorch 1.0.1
* torchvision
* dominate
* Others

## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/Ha0Tang/SelectionGAN
cd SelectionGAN
cd person_transfer
```

### Data Preperation

We use [OpenPose](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation) to generate keypoints. We also provide the prepared images for convience.

#### Market1501
```bash
sh datasets/download_selectiongan_dataset.sh market_data
```

#### DeepFashion
```bash
sh datasets/download_selectiongan_dataset.sh fashion_data
```

### Training
Market-1501
```bash
sh train_market.sh
```

DeepFashion
```bash
sh train_fashion.sh
```

### Testing
Market1501
```bash
sh test_market.sh
```
DeepFashion
```bash
sh test_fashion.sh
```

### Pretrained Models
Market1501
```bash
sh scripts/download_selectiongan_model.sh market
```

DeepFashion
```bash
sh scripts/download_selectiongan_model.sh fashion
```

### Evaluation
Follow [Pose-Transfer](https://github.com/tengteng95/Pose-Transfer) for more details.
