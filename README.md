# Embodied-R: Collaborative Framework for Activating Embodied Spatial Reasoning in Foundation Models via Reinforcement Learning

<a href='https://arxiv.org/pdf/2504.12680'><img src='https://img.shields.io/badge/arXiv-2504.12680-b31b1b.svg'></a> &nbsp;
<a href='https://embodiedcity.github.io/Embodied-R/'><img src='https://img.shields.io/badge/Project-Website-0078D4.svg'></a> &nbsp;
<a href='https://huggingface.co/EmbodiedCity/Embodied-R'><img src='https://img.shields.io/badge/%20Hugging%20Face-Model weights-ffca28.svg'></a>

<p align="center">
  <img src="assets/cover.gif" alt="Cover" width="65%" />
</p>

This project provides the official code for Embodied-R, a collaborative framework designed to enhance embodied spatial reasoning tasks. Embodied-R leverages the perceptual capabilities of large-scale Vision-Language Models (VLMs) and achieves significant performance improvements by training only a small-scale Language Model (LM). By combining the strengths of these models, Embodied-R offers an efficient yet powerful solution for complex spatial reasoning tasks in embodied AI.

<p align="center">
  <img src="assets/framework.png" alt="Framework" width="65%" />
  <img src="assets/rader.png" alt="Results" width="30%" />
</p>

## News

[2025/04/19] We release the basic training and inference code of Embodied-R.

[2025/04/26] We add support for 5-GPU training and local API service, eliminating the need for commercial API calls during training.

[2025/05/13] We uploaded the model weights from two separate training sessions.  [Weight 1] incorporated consistency rewards at a later stage, while [Weight 2] introduced consistency rewards at an earlier stage.
## Installation

The Embodied-R project is built on the ModelScope ms-swift open-source framework. Please follow these steps to install:

1. Ensure your environment meets the following requirements:

   - Python = 3.10
   - Transformers = 4.51
   - DeepSpeed = 0.14.5
   - VLLM = 0.7.3
2. Install the ms-swift framework:

   ```bash
   pip install ms-swift -U
   ```
3. Clone this repository:

   ```bash
   git clone https://github.com/EmbodiedCity/Embodied-R.code.git
   cd Embodied-R.code
   ```

## Setup

### Data Preparation

First, download the [UrbanVideo-Bench](https://huggingface.co/datasets/EmbodiedCity/UrbanVideo-Bench) and [VSI-Bench](https://huggingface.co/datasets/nyu-visionx/VSI-Bench) datasets.

After downloading, organize the directories as shown below (some parts are omitted with "..."):

```
Embodied-R.code/
├── assets/
├── dataset/
│   ├── UrbanVideo-Bench/
│   │   ├── videos/
│   │   ├── MCQ.parquet
│   │   └── ...
│   ├── VSI-Bench/
│   │   ├── arkitscenes.zip
│   │   ├── scannet.zip
│   │   ├── scannetpp.zip
│   │   ├── test-00000-of-00001.parquet
│   │   └── ...
└── ...
```

Then, run the following command and the processed datasets will be stored in `dataset/complete`:

```bash
python dataset/data_preprocess.py
```

After execution, a `videos` folder and three JSON files (`train_data.json`, `val_data.json`, and `test_data.json`) will be generated in the `dataset/complete` directory. The `videos` folder contains the utilized video files. The JSON files follow the format below:

```json
[
    {
        "Question_id": 1,
        "video_id": "EmbodiedCity_20.mp4",
        "question_category": "Counterfactual",
        "question": "Question: Instead of taking a steady descent towards the balcony on the 13th floor, if you choose to hover around the cylindrical building instead, can you complete the task, and how is the alternative route?  \nChoices:  \nA. If I choose to hover around the building, I can complete the task because I avoid obstacles, and the hypothetical movement takes shorter time.  \nB. If I choose to hover around the building, I can complete the task because I acquire a better vantage point, but the alternative takes longer time.  \nC. If I choose to hover around the building, I cannot complete the task because the balcony is only accessible by descending further.  ",
        "answer": "C",
        "scene_name": "EmbodiedCity_20"
    },
    {
        "Question_id": 2,
        "video_id": "EmbodiedCity_38.mp4",
        "question_category": "Counterfactual",
        "question": "Question: Instead of moving forward, if you choose to fly directly to the right after descending alongside the building block, can you still reach the transformer box next to the gate?  \nChoices:  \nA. If I fly directly to the right, I cannot complete the task because I miss the path around the building, and the alternative takes longer.  \nB. If I fly directly to the right, I can complete the task by navigating a shortcut path through the open area, and the alternative takes shorter time.  \nC. If I fly directly to the right, I might not be able to complete the task because I move away from the electric gate on the side, and the hypothetical movement takes longer.  ",
        "answer": "C",
        "scene_name": "EmbodiedCity_38"
    },
    ...
]
```

**If you need to use your own data, please organize it in the format described above.**

### Model Weight Download

Embodied-R uses two main models: a vision module and a reasoning module.

1. **Vision Module Model**:

   - Download [Qwen/Qwen2.5-VL-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct)
   - This large vision-language model is responsible for processing video frames and extracting key semantic information
2. **Reasoning Module Model**:

   - Download [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
   - This small language model is trained with reinforcement learning, specifically for spatial reasoning tasks

   > **Note**: Although the input here is textual, we recommend using the LM Decoder (**Qwen2.5-3B**) in the Qwen2.5-VL-3B-Instruct as the small-scale foundation model. This is because the pretraining of VL models involves multimodal/video-related content, which can benefit the LM Decoder. Fine-tuning on this basis will enable faster convergence.

After downloading, place the model weights in an appropriate directory, or specify the model path when running scripts.

## Inference Examples

Embodied-R provides two inference methods: batch inference and interactive inference.

### Batch Inference

**Important: Complete Video Processing Pipeline**

Before running batch inference, you need to first process videos using `train/conver_format/VLM_perception_local.py` or `train/conver_format/VLM_perception_API.py` to generate text descriptions of the videos. This step converts video content into text representations for the reasoning model to use. The complete pipeline is as follows:

 **Option 1: Using local large model**

   ```bash
   python train/conver_format/VLM_perception_local.py --data_paths [JSON_FILES] --folder_path [VIDEOS_FOLDER] --model_path [VISION_MODEL_PATH] --save_path [RESULTS_PATH]
   ```

   Parameters:
   - `--data_paths`: JSON files containing data, can specify multiple files, default: `['dataset/complete/test_data.json', 'dataset/complete/train_data.json', 'dataset/complete/val_data.json']`
   - `--folder_path`: Folder containing video files, default: `dataset/complete/videos`
   - `--model_path`: Path to the vision model, default: `Qwen/Qwen2.5-Vl-72B-instruct`
   - `--save_path`: Path to save results, default: `results/inter`

   Example (using custom data):
   ```bash
   python train/conver_format/VLM_perception_local.py --data_paths my_data.json --folder_path my_videos_path --model_path Qwen/Qwen2.5-Vl-72B-instruct --save_path my_results_path
   ```

   **Option 2: Using commercial API**

   ```bash
   python train/conver_format/VLM_perception_API.py --data_paths [JSON_FILES] --folder_path [VIDEOS_FOLDER] --api_key [API_KEY] --base_url [API_BASE_URL] --save_path [RESULTS_PATH]
   ```

   Parameters:
   - `--data_paths`: JSON files containing data, can specify multiple files, default: `['dataset/complete/test_data.json', 'dataset/complete/train_data.json', 'dataset/complete/val_data.json']`
   - `--folder_path`: Folder containing video files, default: `dataset/complete/videos`
   - `--api_key`: OpenAI API key, default: None (if not provided, will try to get from environment variable `OPENAI_API_KEY`)
   - `--base_url`: API base URL, default: `https://dashscope.aliyuncs.com/compatible-mode/v1`
   - `--save_path`: Path to save results, default: `results/inter`

   Example (using custom data):
   ```bash
   python train/conver_format/VLM_perception_API.py --data_paths my_data.json --folder_path my_videos_path --api_key your_api_key --save_path my_results_path
   ```

   > **Note**: Qwen officially provides API services for their open-source models, which are identical to the locally deployed small-scale foundation models. If local computing resources are limited, API can be used for training-free reference models.

2. Run batch inference using the generated text descriptions:

   ```bash
   cd infer
   bash run_batch_inference.sh \
     --model "Qwen/Qwen2.5-VL-3B-Instruct" \
     --input_file "results/inter/test_data.json" \
     --output_file "results/infer/inference_result.json" \
     --batch_size 1 \
     --max_tokens 3096
   ```

Input JSON file format example:

```json
[
  {
    "Question_id": "video_infer",
    "video_id": "example.mp4",
    "question_category": "object_rel_direction",
    "question": "<video>Please assume the role of an agent...",
    "answer": "A",
    "videos": "path/to/video.mp4"
  },
  {
    "Question_id": "text_infer",
    "question": "Please assume the role of an agent...",
    "answer": "B"
  }
]
```

**Important Notes**:

- **Video Inference**: You must add the `<video>` prefix to the `question` field and include both `videos` and `question` fields. Other fields (such as `Question_id`, `video_id`, etc.) are optional.
- **Text Inference**: Only the `question` field is required.
- The inference results will preserve all input fields (pass-through) and add a `content` field containing the model's response.

### Interactive Inference

Interactive inference provides a command-line interface that allows users to upload videos and ask questions. Start interactive inference using the following command:

```bash
cd infer
bash run_video_chat.sh
```

You can customize the vision model and reasoning model by modifying the `run_video_chat.sh` script:

```bash
# Set model paths
VISION_MODEL="Qwen/Qwen2.5-VL-72B-Instruct"  # Vision model path
REASONING_MODEL="Qwen/Qwen2.5-VL-3B-Instruct"   # Reasoning model path

# Set parameters
MAX_TOKENS=4096                # Max output tokens for reasoning module
TEMPERATURE=0.7                # Temperature for reasoning module
VISION_MAX_TOKENS=6144         # Max output tokens for vision module
VISION_TEMPERATURE=0.1         # Temperature for vision module
```

## RL Training

Embodied-R uses Reinforcement Learning (RL) to train the reasoning module for high-quality spatial reasoning. The training code is located in the `train` folder.

### Training Environment Requirements

Recommended configurations:

- Standard version: 8x NVIDIA A800 GPUs with 40GB memory each
- Lightweight version: 5x NVIDIA A800 GPUs with 40GB memory each (new)
  - GPUs 0-3: For GRPO training (4-card parallel)
  - GPU 4: For local consistency verification model service

### Training Pipeline

**Important: Complete Training Data Preparation Process**

Before training the model, you need to complete the following data preparation steps:

1. Generate video descriptions using the vision model:

   **Option 1: Using local large model**

   ```bash
   python train/conver_format/VLM_perception_local.py
   ```

   **Option 2: Using commercial API**

   ```bash
   python train/conver_format/VLM_perception_API.py
   ```
2. Convert the generated text descriptions to GRPO training format:

   ```bash
   python train/conver_format/convert_GrpoFormat.py
   ```
3. Start training:

   **New 5-GPU version** (uses local API service for consistency reward):

   ```bash
   bash train/train_5GPUs.sh
   ```

   **Standard 8-GPU version** (uses commercial API for consistency reward):

   ```bash
   bash train/train_8GPUs.sh
   ```

The training script uses the GRPO (Group Relative Policy Optimization) algorithm, a PPO variant designed specifically for large language models. You can customize the training process by modifying parameters in the training scripts:

```bash
# Key parameters in both scripts
--model "Qwen/Qwen2.5-VL-3B-Instruct"  # Base model
--reward_weights 0.7 0.1 0.2         # Reward weights (accuracy, format, consistency)
--reward_funcs choice_accuracy format consistency  # Reward functions
--learning_rate 5e-7                 # Learning rate
--num_train_epochs 2                 # Number of training epochs
```

**Key differences between the two versions:**

1. **5-GPU version** (`train_5GPUs.sh`):

   - Uses 4 GPUs (0-3) for training
   - Uses 1 GPU (4) for local consistency verification service
   - Automatically starts the local consistency service
   - Uses local API for consistency reward (`consistency_reward_local.py`)
   - No need for commercial API keys

2. **8-GPU version** (`train_8GPUs.sh`):

   - Uses all 8 GPUs for training
   - Uses commercial API for consistency reward (`consistency_reward_API.py`)
   - Higher throughput with more GPUs

For more details about the local consistency service, please refer to `train/reward/README_local_consistency.md`.

### Reward Modeling

Embodied-R uses two main rewards to guide model learning:

1. **Choice Accuracy Reward**:

   - Evaluates whether the model's answer matches the correct answer
   - Implemented in `train/reward/choice_accuracy_reward.py`
2. **Format Reward**:
   - Ensures the model output follows the format `<think>reasoning process</think><answer>answer</answer>`
3. **Consistency Reward**:

   - Evaluates whether the model's reasoning process is logically consistent with its final answer
   - Works by inputting the reasoning process into a reference model to check if it produces the same answer
   - **Two options for reference model access**:

     a) **Local API Service**:

     - Implemented in `train/reward/consistency_reward_local.py`
     - Used in the 5-GPU version (`train_5GPUs.sh`)
     - Runs a local model service on GPU 4

     ```bash
     # Start the local API service
     bash train/reward/start_consistency_service.sh
     ```

     b) **Commercial API (Bailian platform)**:

     - Implemented in `train/reward/consistency_reward_API.py`
     - Used in the 8-GPU version (`train_8GPUs.sh`)

     ```python
     # Enter your API keys here
     default_api_keys = [
         # API keys obtained from the Bailian platform
     ]
     ```

     Please visit the [Bailian platform](https://www.aliyun.com/product/bailian) to apply for API keys

We recommend using different reward weight ratios at different stages of training. Overall, higher weight should be assigned to the format reward in the early stages. In later stages, greater emphasis should be placed on the accuracy reward, while gradually incorporating the logical consistency reward. The earlier the logical consistency reward is introduced, the better the model's logical coherence can be maintained, though this may lead to slower training progress. Please adjust accordingly.

## Citation

```
@inproceedings{10.1145/3746027.3755703,
  author = {Zhao, Baining and Wang, Ziyou and Fang, Jianjie and Gao, Chen and Man, Fanhang and Cui, Jinqiang and Wang, Xin and Chen, Xinlei and Li, Yong and Zhu, Wenwu},
  title = {Embodied-R: Collaborative Framework for Activating Embodied Spatial Reasoning in Foundation Models via Reinforcement Learning},
  year = {2025},
  isbn = {9798400720352},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3746027.3755703},
  doi = {10.1145/3746027.3755703},
  booktitle = {Proceedings of the 33rd ACM International Conference on Multimedia},
  pages = {11071–11080},
  numpages = {10},
  keywords = {embodied intelligence, reasoning, vision-language model},
  location = {Dublin, Ireland},
  series = {MM '25}
}
```
