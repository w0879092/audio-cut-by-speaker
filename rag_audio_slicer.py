import argparse
import os
# 必须放在 import torch 之前
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import json
import gc
import subprocess
import numpy as np
import torch
import librosa
import time
import shutil
import re  # 🌟 新增：用于解析 FFmpeg 静音探测日志
from tqdm import tqdm
from pyannote.audio import Pipeline, Model
from pyannote.audio.core.inference import Inference
from pyannote.audio.pipelines.utils.hook import ProgressHook
from scipy.spatial.distance import cosine

# ================= 🌟 核心保命符 =================
torch.backends.cudnn.enabled = False 

# ================= 1. 命令行交互设计 =================
parser = argparse.ArgumentParser(description="🤖 祷告塔 AI 智能分布式声纹切割引擎")
parser.add_argument("audio_file", type=str, help="必须提供: 要处理的音频文件完整路径 (例如 /mnt/z/.../xxx.m4a)")
args = parser.parse_args()

# ================= 2. 核心参数与集群路径配置 =================
remote_audio_file = args.audio_file
remote_output_base_dir = "/mnt/z/祷告塔文件分割"

# 🌍 分布式声纹库云端路径
REMOTE_DB_PATH = "/mnt/z/祷告塔文件分割/声纹库/speaker_db_main.json"
LOCAL_DB_PATH = "speaker_db.json"

local_temp_workspace = "./Local_AI_Workspace"
local_temp_input = os.path.join(local_temp_workspace, "temp_input")
local_temp_output = os.path.join(local_temp_workspace, "temp_output")

HF_TOKEN = " "

MONOLOGUE_THRESHOLD = 180.0
MAX_PAUSE_MERGE = 5.0
MAX_DIALOGUE_GAP = 30.0

# 📈 工业级声纹标准参数
MIN_QUERY_DURATION = 3.0      # 认人底线：最长发言 > 3.0秒 才去数据库比对
MIN_ENROLL_DURATION = 8.0     # 建档底线：最长发言 > 8.0秒 才有资格写入主脑数据库
SIMILARITY_THRESHOLD = 0.82   # 严苛的认人及格线 (防误判)
LEARNING_THRESHOLD = 0.90     # 自进化学习线 (防过度重复死记硬背)
MAX_VECTORS_PER_PERSON = 15   # 每人最多保留10个高清变异音色
MAX_ENROLL_PAUSE = 2          # 允许缝合的最大停顿时间
SILENCE_DB_THRESHOLD = -35.0  # 🌟 新增：物理静音闸门，拦截幽灵说话人
MIN_DIALOGUE_DURATION = 10.0  # 🌟 新增：物理输出最低底线，拦截孤岛碎片
SPLIT_THRESHOLD = 4 * 3600    # 🌟 新增：超长音频切分阈值 (4小时)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
time_records = {}
total_start_time = time.time()

# ================= 🌟 新增：分治切割辅助函数 =================
def find_best_split_point(audio_path, target_hour=5, window_minutes=10):
    target_time = target_hour * 3600
    window_half = (window_minutes * 60) / 2
    start_search = max(0, target_time - window_half)
    cmd = [
        "ffmpeg", "-y", "-hide_banner",
        "-ss", str(start_search),
        "-t", str(window_minutes * 60),
        "-i", audio_path,
        "-af", "silencedetect=noise=-35dB:d=1.5",
        "-f", "null", "-"
    ]
    result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
    silence_starts = re.findall(r'silence_start: ([\d\.]+)', result.stderr)
    if not silence_starts:
        return target_time
    return float(silence_starts[0]) + start_search

def split_large_audio(audio_path, split_time, output_part1, output_part2):
    print(f"✂️ 正在执行极速无损分段，安全下刀点：{split_time:.2f} 秒处...")
    cmd1 = ["ffmpeg", "-y", "-t", str(split_time), "-i", audio_path, "-c", "copy", output_part1]
    subprocess.run(cmd1, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    cmd2 = ["ffmpeg", "-y", "-ss", str(split_time), "-i", audio_path, "-c", "copy", output_part2]
    subprocess.run(cmd2, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# 🌟 第一阶段：全局数据提纯与预处理
def preprocess_and_purify(raw_turns):
    speaker_durations = {}
    for turn in raw_turns:
        spk = turn["speaker"]
        dur = turn["end"] - turn["start"]
        speaker_durations[spk] = speaker_durations.get(spk, 0.0) + dur
        
    valid_turns = []
    for turn in raw_turns:
        if speaker_durations[turn["speaker"]] >= 10.0:
            valid_turns.append(turn.copy())
            
    if not valid_turns:
        return []
        
    valid_turns.sort(key=lambda x: x["start"])
    
    merged_turns = []
    for turn in valid_turns:
        if not merged_turns:
            merged_turns.append(turn)
            continue
            
        prev_turn = merged_turns[-1]
        if turn["speaker"] == prev_turn["speaker"] and (turn["start"] - prev_turn["end"]) <= 1.0:
            prev_turn["end"] = max(prev_turn["end"], turn["end"])
        else:
            merged_turns.append(turn)
            
    for turn in merged_turns:
        turn["start"] = max(0.0, turn["start"] - 0.2)
        turn["end"] = turn["end"] + 0.2
        
    return merged_turns

# ================= 3. 🌍 分布式云端同步引擎 =================
def get_next_speaker_id(db):
    max_num = 0
    for data in db.values():
        if "id" in data and data["id"].startswith("A"):
            try:
                num = int(data["id"][1:])
                if num > max_num: max_num = num
            except ValueError:
                pass
    return f"A{max_num + 1:04d}"

def load_json_db(path):
    if not os.path.exists(path): return {}
    with open(path, "r", encoding="utf-8") as f:
        try: db = json.load(f)
        except: return {}
        
        needs_save = False
        for name, data in db.items():
            if "vector" in data: 
                data["vectors"] = [data.pop("vector")]
                needs_save = True
                
        for name, data in db.items():
            if "id" not in data:
                data["id"] = get_next_speaker_id(db)
                needs_save = True
                
        if needs_save:
            save_json_db(db, path)
        return db

def save_json_db(db, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=4)

def identify_speaker(new_vector, current_db):
    best_match, highest_sim = None, -1
    for name, data in current_db.items():
        for db_vector in data.get("vectors", []):
            sim = 1 - cosine(new_vector, db_vector)
            if sim > highest_sim:
                highest_sim, best_match = sim, name
    return (best_match, highest_sim) if highest_sim >= SIMILARITY_THRESHOLD else (None, highest_sim)

def sync_and_merge_database():
    print("\n🔄 [云端同步] 正在校验本地副库与 NAS 主库...")
    os.makedirs(os.path.dirname(REMOTE_DB_PATH), exist_ok=True)
        
    has_remote = os.path.exists(REMOTE_DB_PATH)
    has_local = os.path.exists(LOCAL_DB_PATH)
    
    if not has_remote and not has_local:
        print("  ℹ️ 初始状态：双端均无数据库，将自动创建。")
        return {}
        
    if not has_remote and has_local:
        print("  ⬆️ 正在将本地库初始化为 NAS 主库...")
        shutil.copy2(LOCAL_DB_PATH, REMOTE_DB_PATH)
        return load_json_db(LOCAL_DB_PATH)
        
    if has_remote and not has_local:
        print("  ⬇️ 正在拉取 NAS 主库到本地...")
        shutil.copy2(REMOTE_DB_PATH, LOCAL_DB_PATH)
        return load_json_db(LOCAL_DB_PATH)
        
    base_db = load_json_db(REMOTE_DB_PATH)
    incoming_db = load_json_db(LOCAL_DB_PATH)
    
    if json.dumps(base_db, sort_keys=True) == json.dumps(incoming_db, sort_keys=True):
        print("  ✅ 数据库已是最新同步状态。")
        return base_db
        
    print("  ⚠️ 发现双端数据差异，正在将本地新数据融合至 NAS 主脑...")
    merge_stats = {"matched": 0, "new_speakers": 0}
    
    for inc_name, inc_data in incoming_db.items():
        for inc_vector in inc_data.get("vectors", []):
            best_match, highest_sim = identify_speaker(inc_vector, base_db)
                        
            if highest_sim >= SIMILARITY_THRESHOLD:
                if len(base_db[best_match]["vectors"]) < MAX_VECTORS_PER_PERSON:
                    is_duplicate = any((1 - cosine(inc_vector, bv)) > 0.98 for bv in base_db[best_match]["vectors"])
                    if not is_duplicate:
                        base_db[best_match]["vectors"].append(inc_vector)
                        merge_stats["matched"] += 1
            else:
                if not inc_name.startswith("Voice_") and inc_name not in base_db:
                    new_id = inc_name
                else:
                    base_name = inc_name + "_local"
                    new_id = base_name
                    idx = 1
                    while new_id in base_db:
                        new_id = f"{base_name}_{idx}"
                        idx += 1
                
                new_a_id = get_next_speaker_id(base_db)
                base_db[new_id] = {"id": new_a_id, "vectors": [inc_vector]}
                merge_stats["new_speakers"] += 1
                
    print(f"  🔗 融合完成: 吸收新音色 {merge_stats['matched']} 个，新增成员 {merge_stats['new_speakers']} 个")
                
    save_json_db(base_db, REMOTE_DB_PATH)
    shutil.copy2(REMOTE_DB_PATH, LOCAL_DB_PATH)
    print("  ✅ 最新权威声纹库已下载至本地！")
    return base_db

# ================= 4. 分布式工作流初始化 =================
print(f"🖥️ 当前计算设备: {device} (AMD ROCm 模式)")
t_sync = time.time()
speaker_db = sync_and_merge_database()
time_records['0. 分布式云端数据库同步'] = time.time() - t_sync

# ================= 5. 拉取文件到本地工作空间 =================
file_basename = os.path.splitext(os.path.basename(remote_audio_file))[0]
file_ext = os.path.splitext(remote_audio_file)[1]

os.makedirs(local_temp_input, exist_ok=True)
os.makedirs(local_temp_output, exist_ok=True)
local_audio_path = os.path.join(local_temp_input, f"local_working{file_ext}")

print(f"\n⬇️ [预备阶段] 正在将 NAS 文件拉取至本地缓存...")
t_pull = time.time()
if not os.path.exists(remote_audio_file):
    raise FileNotFoundError(f"❌ 找不到输入文件: {remote_audio_file}")

shutil.copy2(remote_audio_file, local_audio_path)
time_records['1. NAS文件拉取'] = time.time() - t_pull
print(f"✅ 文件已缓存！准备进入 AI 核心计算环节。\n")

# ================= 🌟 新增：封装原有的 6-10 步骤，以便分治调用 =================
def process_audio_chunk(chunk_path, time_offset=0.0, global_index_start=0, chunk_label=""):
    global pending_start_time
    pending_start_time = None
    
    # ================= 6. 加载本地音频特征 =================
    print(f"\n🎵 正在读取音频特征 {chunk_label}...")
    t0 = time.time()
    wav_np, sample_rate = librosa.load(chunk_path, sr=16000, mono=True)
    waveform = torch.from_numpy(wav_np).unsqueeze(0).to(torch.float32)
    time_records[f'2. 本地音频加载 {chunk_label}'] = time.time() - t0

    # ================= 7. AI 声纹分离 (步骤 1/4) =================
    print(f"\n👥 [步骤 1/4] 正在分析说话人时间轴 (Diarization) {chunk_label}...")
    t1 = time.time()
    
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=HF_TOKEN).to(device)
    
    with ProgressHook() as hook:
        diarization_result = pipeline({"waveform": waveform, "sample_rate": sample_rate}, hook=hook)
    
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    time_records[f'3. AI时间轴分离 {chunk_label}'] = time.time() - t1

    annotation = diarization_result.speaker_diarization if hasattr(diarization_result, "speaker_diarization") else diarization_result

    # ================= 8. 提取指纹与声纹比对 (步骤 2/4) =================
    print(f"\n🧬 [步骤 2/4] 正在提取指纹 (CPU 极致稳健模式) {chunk_label}...")
    t_load_emb = time.time()
    
    embedding_model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM", token=HF_TOKEN).to("cpu")
    inference = Inference(embedding_model, window="whole") 
    
    time_records[f'4. 声纹模型加载 {chunk_label}'] = time.time() - t_load_emb

    t2 = time.time()
    local_speaker_turns = {}
    raw_turns = []
    golden_segments_to_cut = [] 
    
    for turn, _, spk in annotation.itertracks(yield_label=True):
        raw_turns.append({"start": turn.start, "end": turn.end, "speaker": spk})

    purified_turns = preprocess_and_purify(raw_turns)

    for turn in purified_turns:
        spk = turn["speaker"]
        if spk not in local_speaker_turns: 
            local_speaker_turns[spk] = []
        local_speaker_turns[spk].append((turn["start"], turn["end"]))

    speaker_mapping = {}
    for spk, turns in tqdm(local_speaker_turns.items(), desc=f"执行漏斗比对 {chunk_label}", unit="人"):
        turns.sort()
        fused_turns = []
        if turns:
            curr_start, curr_end = turns[0]
            for i in range(1, len(turns)):
                next_start, next_end = turns[i]
                if next_start - curr_end <= MAX_ENROLL_PAUSE:
                    curr_end = max(curr_end, next_end)
                else:
                    fused_turns.append((curr_start, curr_end))
                    curr_start, curr_end = next_start, next_end
            fused_turns.append((curr_start, curr_end))
        
        longest_fused_turn = max(fused_turns, key=lambda x: x[1] - x[0])
        f_start, f_end = longest_fused_turn
        
        overlapping_others = sorted([t for t in raw_turns if t["speaker"] != spk and t["start"] < f_end and t["end"] > f_start], key=lambda x: x["start"])
        
        current_pure_start = f_start
        pure_fragments = []
        for ot in overlapping_others:
            if ot["start"] > current_pure_start:
                pure_fragments.append((current_pure_start, ot["start"]))
            current_pure_start = max(current_pure_start, ot["end"])
        if current_pure_start < f_end:
            pure_fragments.append((current_pure_start, f_end))
            
        pure_fragments = [f for f in pure_fragments if f[1] - f[0] >= 1.0]
        
        if not pure_fragments:
            speaker_mapping[spk] = spk 
            tqdm.write(f"  🔇 忽略 {spk}: 剔除重叠污染后无纯净片段，不予识别。")
            continue

        longest_pure_frag = max(pure_fragments, key=lambda x: x[1] - x[0])
        longest_pure_duration = longest_pure_frag[1] - longest_pure_frag[0]
        total_pure_duration = sum(f[1] - f[0] for f in pure_fragments)
        
        is_concatenated = False
        crops = []
        golden_segments = []
        
        if longest_pure_duration >= MIN_ENROLL_DURATION:
            max_duration = longest_pure_duration
            g_start, g_end = longest_pure_frag
            g_end = min(g_start + 60.0, g_end)
            crop = waveform[:, int(g_start * sample_rate):int(g_end * sample_rate)]
            golden_segments.append((g_start, g_end))
        else:
            max_duration = total_pure_duration
            if max_duration >= MIN_QUERY_DURATION:
                is_concatenated = True
                acc_duration = 0.0
                for f_s, f_e in pure_fragments:
                    dur = f_e - f_s
                    if acc_duration + dur > 60.0:
                        f_e = f_s + (60.0 - acc_duration)
                        dur = f_e - f_s
                    crops.append(waveform[:, int(f_s * sample_rate):int(f_e * sample_rate)])
                    golden_segments.append((f_s, f_e))
                    acc_duration += dur
                    if acc_duration >= 60.0:
                        break
                crop = torch.cat(crops, dim=1)

        rms = torch.sqrt(torch.mean(crop ** 2))
        dbfs = 20 * torch.log10(rms + 1e-9).item()
        
        if dbfs < SILENCE_DB_THRESHOLD:
            speaker_mapping[spk] = "GHOST_SILENCE"
            tqdm.write(f"  👻 抹杀幽灵 {spk}: 音量 {dbfs:.1f}dB 低于闸门，全网除名。")
            continue
        
        if max_duration < MIN_QUERY_DURATION:
            speaker_mapping[spk] = spk 
            tqdm.write(f"  🔇 忽略 {spk}: 最长连续发言仅 {max_duration:.1f}s，不予识别。")
            continue
            
        vector = inference({"waveform": crop, "sample_rate": sample_rate}).tolist()
        matched_name, sim = identify_speaker(vector, speaker_db)
        
        if matched_name:
            tqdm.write(f"  🔍 匹配 {spk} -> 【{matched_name}】 (相似度: {sim:.2f}, 采样时长: {max_duration:.1f}s)")
            speaker_mapping[spk] = matched_name
            
            if max_duration >= MIN_ENROLL_DURATION:
                if sim < LEARNING_THRESHOLD and len(speaker_db[matched_name]["vectors"]) < MAX_VECTORS_PER_PERSON:
                    speaker_db[matched_name]["vectors"].append(vector)
                    tqdm.write(f"  🧠 AI已安全吸收【{matched_name}】的全新音色特征。")
                    golden_segments_to_cut.append({"speaker": matched_name, "segments": golden_segments, "is_concatenated": is_concatenated})
        else:
            if max_duration >= MIN_ENROLL_DURATION:
                new_id = f"Voice_{len(speaker_db) + 1:03d}"
                tqdm.write(f"  🆕 建档 {spk} -> 录入主脑数据库【{new_id}】 (采样时长: {max_duration:.1f}s)")
                speaker_db[new_id] = {"vectors": [vector]}
                speaker_mapping[spk] = new_id
                golden_segments_to_cut.append({"speaker": new_id, "segments": golden_segments, "is_concatenated": is_concatenated})
            else:
                speaker_mapping[spk] = spk
                tqdm.write(f"  ⚠️ 拒收 {spk}: 发现新声音，但最长发言仅 {max_duration:.1f}s，未达建档线。")

    save_json_db(speaker_db, LOCAL_DB_PATH)
    
    # 🌟 物理清场机制：提前释放，杜绝 fork 死锁
    del embedding_model, inference, waveform, wav_np
    gc.collect()
    torch.cuda.empty_cache()
    time_records[f'5. 指纹提取与自进化 {chunk_label}'] = time.time() - t2

    # ================= 🌟 第二阶段：动态静音管理与极短插话吸收 =================
    def apply_dynamic_silence_and_absorption(turns):
        if not turns:
            return []
            
        # 步骤 1：极短插话的静默吸收 (Absorption)
        # 筛选出所有单次发言大于 2 秒的“稳固段落”索引
        solid_indices = [i for i, t in enumerate(turns) if (t['end'] - t['start']) > 2.0]
        
        # 寻找被同一个主讲人包围的插话区
        for i in range(len(solid_indices) - 1):
            left_idx = solid_indices[i]
            right_idx = solid_indices[i+1]
            
            # 如果左右两个稳固段落属于同一个人
            if turns[left_idx]['speaker'] == turns[right_idx]['speaker']:
                main_speaker = turns[left_idx]['speaker']
                # 将夹在中间的所有短片段强行吸收进主讲人的音频流
                for j in range(left_idx + 1, right_idx):
                    turns[j]['speaker'] = main_speaker

        # 步骤 2：动态静音删除与同人合并
        processed_turns = []
        for turn in turns:
            if not processed_turns:
                processed_turns.append(turn.copy())
                continue
                
            prev = processed_turns[-1]
            gap = turn['start'] - prev['end']
            
            if turn['speaker'] == prev['speaker']:
                # 同人静音：<= 60秒，保留静音并缝合；> 60秒，断开并删除多余静音
                if gap <= 60.0:
                    prev['end'] = max(prev['end'], turn['end'])
                else:
                    processed_turns.append(turn.copy())
            else:
                # 跨人静音：<= 5秒，算作前一个人的尾巴（填补物理静音区间）
                if 0 < gap <= 5.0:
                    prev['end'] = turn['start']
                # 无论跨人静音多长，只要换人了，就产生新的独立分段
                processed_turns.append(turn.copy())
                
        return processed_turns

    # ================= 9. 计算切割逻辑 (步骤 3/4) =================
    print(f"\n⚙️ [步骤 3/4] 正在计算 3分钟 分割逻辑 {chunk_label}...")
    t3 = time.time()
    
    for t in raw_turns:
        t["speaker"] = speaker_mapping.get(t["speaker"], t["speaker"])
        
    valid_turns = [t for t in raw_turns if t["speaker"] != "GHOST_SILENCE"]
    valid_turns.sort(key=lambda x: x["start"])
    
    # 🌟 调用模块二：执行动态静音管理与极短插话吸收
    managed_turns = apply_dynamic_silence_and_absorption(valid_turns)
    
    # ================= 🌟 第三阶段：密集讨论区判定与长段落排斥算法 =================
    print(f"\n⚙️ [步骤 3/4] 正在计算 密集讨论区与排斥算法 {chunk_label}...")
    t3 = time.time()

    blocks = []
    current_block = []

    for turn in managed_turns:
        curr_dur = turn['end'] - turn['start']
        
        if not current_block:
            current_block.append(turn)
            continue
            
        prev = current_block[-1]
        prev_dur = prev['end'] - prev['start']
        gap = turn['start'] - prev['end']
        
        should_cut = False
        
        # 如果物理间隔大于 5 秒，强制断开新起一段
        if gap > 5.0:
            should_cut = True
        else:
            # 1. 150秒长段落排斥与豁免规则
            if curr_dur > 150.0:
                long_turns = [t for t in current_block if (t['end'] - t['start']) > 150.0]
                if long_turns:
                    last_long = long_turns[-1]
                    # 豁免规则判定：判断上一个长段落是否为同一人
                    if last_long['speaker'] == turn['speaker']:
                        idx = current_block.index(last_long)
                        intermediate = current_block[idx+1:]
                        # 检查中间夹杂的是否全部为 <= 2秒的短片段
                        if all((t['end'] - t['start']) <= 2.0 for t in intermediate):
                            should_cut = False # 🌟 触发豁免：维持合并
                        else:
                            should_cut = True  # 排斥生效
                    else:
                        should_cut = True      # 非同人，排斥生效
            
            # 2. 密集讨论区判定 (必须双方均 > 2秒才合并，否则强制断开)
            if not should_cut:
                if prev['speaker'] != turn['speaker']:
                    if prev_dur <= 2.0 or curr_dur <= 2.0:
                        should_cut = True

        if should_cut:
            blocks.append(current_block)
            current_block = [turn]
        else:
            current_block.append(turn)
            
    if current_block:
        blocks.append(current_block)

    # ================= 🌟 第四阶段：最终产出标注逻辑 =================
    output_segments = []
    for block in blocks:
        block_start = block[0]['start']
        block_end = block[-1]['end']
        total_dur = sum(t['end'] - t['start'] for t in block)
        
        # 统计块内各说话人时长占比
        spk_durations = {}
        for t in block:
            spk_durations[t['speaker']] = spk_durations.get(t['speaker'], 0.0) + (t['end'] - t['start'])
            
        max_speaker = max(spk_durations, key=spk_durations.get)
        max_ratio = spk_durations[max_speaker] / total_dur if total_dur > 0 else 0
        
        if len(block) == 1:
            mode_str = "单人发言"
            speaker_id = block[0]['speaker']
        else:
            if max_ratio > 0.80:
                mode_str = "主讲人被插话"
                speaker_id = max_speaker
            else:
                mode_str = "多人对话"
                # 多人对话模式下，取第一个开口人的名字。
                # 遵循你“说话人标识放最前面”和“采用多人对话-名字”的双重指令组合。
                speaker_id = block[0]['speaker']
                
        output_segments.append({
            "start": block_start,
            "end": block_end,
            "speaker": speaker_id,
            "mode": mode_str
        })
        
    time_records[f'6. 核心切割逻辑计算 {chunk_label}'] = time.time() - t3

    # ================= 10. FFmpeg 本地无损切割 (步骤 4/4) =================
    print(f"\n✂️ [步骤 4/4] 正在本地进行极速无损切割，共 {len(output_segments)} 段 {chunk_label}...")
    t4 = time.time()
    
    for seg_info in tqdm(golden_segments_to_cut, desc=f"提取黄金声纹片段 {chunk_label}", unit="段"):
        feature_filename = f"{seg_info['speaker']}说话人特征{file_ext}"
        feature_filepath = os.path.join(local_temp_output, feature_filename)
        
        if not seg_info["is_concatenated"] or len(seg_info["segments"]) == 1:
            g_start, g_end = seg_info["segments"][0]
            cmd = ["ffmpeg", "-y", "-ss", str(g_start), "-t", str(g_end - g_start),
                   "-i", chunk_path, "-c", "copy", "-map_metadata", "0", feature_filepath]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            filter_str = ""
            for idx, (g_start, g_end) in enumerate(seg_info["segments"]):
                filter_str += f"[0:a]atrim=start={g_start}:end={g_end},asetpts=PTS-STARTPTS[aud{idx}];"
            
            concat_inputs = "".join([f"[aud{idx}]" for idx in range(len(seg_info["segments"]))])
            filter_str += f"{concat_inputs}concat=n={len(seg_info['segments'])}:v=0:a=1[outa]"
            
            cmd_concat = [
                "ffmpeg", "-y", "-i", chunk_path,
                "-filter_complex", filter_str,
                "-map", "[outa]",
                feature_filepath
            ]
            subprocess.run(cmd_concat, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for i, seg in enumerate(tqdm(output_segments, desc=f"物理切割音频 {chunk_label}", unit="段")):
        # 🌟 严格落实极简命名逻辑，剔除时长，统一标签前置，结合四阶段画像动态命名
        filename = f"{seg['speaker']}_{seg['mode']}_{global_index_start + i + 1:03d}{file_ext}"
        local_filepath = os.path.join(local_temp_output, filename)

        cmd = ["ffmpeg", "-y", "-ss", str(seg['start']), "-t", str(seg['end'] - seg['start']),
               "-i", chunk_path, "-c", "copy", "-map_metadata", "0", local_filepath]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
    time_records[f'7. 本地FFmpeg切割 {chunk_label}'] = time.time() - t4
    
    return len(output_segments)

# ================= 🌟 核心调度逻辑区 =================
try:
    total_duration = librosa.get_duration(filename=local_audio_path)
    
    if total_duration > SPLIT_THRESHOLD:
        print("\n" + "="*45)
        print(f" ⚠️ 检测到超长音频 (约 {total_duration/3600:.1f} 小时)")
        print(f" 触发智能分治调度策略，以防内存死锁")
        print("="*45)
        
        target_hour = (total_duration / 2) / 3600
        split_point = find_best_split_point(local_audio_path, target_hour=target_hour)
        
        part1_path = os.path.join(local_temp_input, f"part1{file_ext}")
        part2_path = os.path.join(local_temp_input, f"part2{file_ext}")
        
        split_large_audio(local_audio_path, split_point, part1_path, part2_path)
        
        print("\n========== 启动上半场处理 ==========")
        count_part1 = process_audio_chunk(part1_path, time_offset=0.0, global_index_start=0, chunk_label="(上半场)")
        
        # 将最新的声纹库强行回写一遍，保证下半场能读到
        save_json_db(speaker_db, LOCAL_DB_PATH) 
        
        print("\n========== 启动下半场处理 ==========")
        process_audio_chunk(part2_path, time_offset=split_point, global_index_start=count_part1, chunk_label="(下半场)")
        
        # 清理分治碎片
        if os.path.exists(part1_path): os.remove(part1_path)
        if os.path.exists(part2_path): os.remove(part2_path)
        
    else:
        print("\n常规音频长度，执行单次处理逻辑...")
        process_audio_chunk(local_audio_path, time_offset=0.0, global_index_start=0, chunk_label="")

    # ================= 11. 将成品推送至 NAS =================
    print(f"\n📤 [收尾阶段] 正在将切割成品安全推送至 NAS...")
    t_push = time.time()
    final_remote_dir = os.path.join(remote_output_base_dir, file_basename)
    os.makedirs(final_remote_dir, exist_ok=True)

    sliced_files = os.listdir(local_temp_output)
    for f in tqdm(sliced_files, desc="推送至 NAS", unit="文件"):
        shutil.copyfile(os.path.join(local_temp_output, f), os.path.join(final_remote_dir, f))
        
    time_records['8. 成品推送至NAS'] = time.time() - t_push
    print(f"✅ 推送成功！文件已存放在: {final_remote_dir}")
    task_success = True

except Exception as e:
    print(f"\n❌ 运行中途发生错误: {e}")
    task_success = False

finally:
    # ================= 12. 格式化性能报表 (XX分XX秒) =================
    def format_time(seconds):
        m, s = divmod(int(seconds), 60)
        return f"{m}分{s}秒"

    if 'task_success' in locals() and task_success:
        print(f"\n🧹 正在清理本地临时缓存...")
        if os.path.exists(local_temp_workspace):
            shutil.rmtree(local_temp_workspace)
        
        print("\n" + "="*45)
        print(" ⏱️ 任务全链路性能统计报告")
        print("="*45)
        for step_name in sorted(time_records.keys()):
            print(f" {step_name:<25} : {format_time(time_records[step_name])}")
        print("-" * 45)
        print(f" 🌟 任务总耗时              : {format_time(time.time() - total_start_time)}")
        print("="*45)
        print(f"\n🎉 完美收工！")
    else:
        print(f"\n⚠️ 警告：任务未完全成功！")
        print(f"📂 切片文件可能残留在本地工作区：{os.path.abspath(local_temp_output)}")
