# 适用于windows 系统 AMD显卡 Ubuntu WSL2环境

## 挂载SMB Z盘作为远程路径

确保挂载点目录存在
sudo mkdir -p /mnt/z

将 Windows 的 Z 盘映射接入 WSL
sudo mount -t drvfs Z: /mnt/z

确认挂载成功

## 激活环境
conda activate whisper_env

## 跑单个文件
python rag_audio_slicer.py "/mnt/z/sample.m4a"(文件路径)

## 跑文件夹
python batch_runner.py /mnt/z/sample (文件夹路径)



