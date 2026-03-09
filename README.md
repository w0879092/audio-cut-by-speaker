# 适用于windows 系统 AMD显卡 Ubuntu WSL2环境

## 挂载SMB Z盘作为远程路径

确保挂载点目录存在
sudo mkdir -p /mnt/z

将 Windows 的 Z 盘映射接入 WSL
sudo mount -t drvfs Z: /mnt/z

确认挂载成功

## 激活环境
conda activate whisper_env

安装最新版 (支持 ROCm 6.3) 的 PyTorch
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.3

1. 安装核心新增依赖 (阿里 ModelScope 与 音频桥接库)
pip install modelscope soundfile

2. 更新 Pyannote 引擎
pip install --upgrade pyannote.audio （慎用！！！很可能毁掉ROCm）

3. 安装其他依赖
pip install addict onnxruntime oss2 pyyaml einops accelerate safetensors transformers pydantic jsonlines jieba simplejson sortedcontainers
pip install "datasets>=2.19.0,<3.0.0"


5. 环境自检
ffmpeg -version | head -n 1

python -c "
import torch
import pyannote.audio
import modelscope
import soundfile

print('\n' + '='*40)
print(' 🚀 祷告塔 AI 引擎 - 环境自检系统')
print('='*40)

print('\n[1] 深度学习与底层硬件')
print(' - PyTorch 核心版本 :', torch.__version__)
print(' - AMD ROCm/HIP 版本:', torch.version.hip)
gpu_status = '🟢 成功激活' if torch.cuda.is_available() else '🔴 致命错误: 未激活'
print(' - GPU 硬件加速状态 :', gpu_status)

if torch.cuda.is_available():
    print(' - 识别到的计算巨兽 :', torch.cuda.get_device_name(0))

print('\n[2] 混合双擎与桥接依赖')
print(' - Pyannote (时间轴引擎) :', pyannote.audio.__version__)
print(' - ModelScope (阿里生态) : 🟢 已成功导入')
print(' - Soundfile (张量桥接)  : 🟢 已成功导入')
print('\n' + '='*40 + '\n')
"

