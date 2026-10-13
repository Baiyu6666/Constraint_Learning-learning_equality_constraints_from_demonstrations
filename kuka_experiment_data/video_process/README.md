# iiwa 双摄实验视频后处理

这是从 `iiwa_DS_docker` 独立出来的离线后处理项目。它不会连接 ROS、FRI 或机器人，只读取已经保存的文件：

- 主摄像头：`*_cam1.mp4`
- 主摄像头时间戳：`*_cam1_timestamps.csv`
- 副摄像头：`*_cam2.mp4`
- 副摄像头时间戳：`*_cam2_timestamps.csv`
- 机器人数据：`constraint_plan_*.json`

输出保持主视频原始分辨率，在右侧叠加副摄像头画面和三组时间对齐的 Profile：position errors、orientation errors 和 end-effector speed。

## 安装

系统需要 `ffmpeg`。Python 依赖可以安装到独立虚拟环境：

```bash
cd /home/baiyu/pc_share/iiwa_dual_camera_postprocess
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 列出可处理录像

在宿主机项目中，当前录像和轨迹目录通常是：

```bash
python process_recording.py \
  --recording-directory /home/baiyu/iiwa_DS_docker/exp_logs/videos \
  --trace-directory /home/baiyu/iiwa_DS_docker/src/iiwa_toolkit/constraint_run_logs \
  list
```

只有双摄视频、两份时间戳和至少一条时间重叠的真实轨迹都存在时，`complete` 才为 `true`。

## 处理最新录像

```bash
python process_recording.py \
  --recording-directory /home/baiyu/iiwa_DS_docker/exp_logs/videos \
  --trace-directory /home/baiyu/iiwa_DS_docker/src/iiwa_toolkit/constraint_run_logs \
  process latest
```

也可以把 `latest` 换成 `list` 显示的 recording ID，并用 `--output /path/result.mp4` 指定输出位置。

## 直接指定文件

如果文件不遵循自动发现的命名规则，可以直接运行底层渲染器：

```bash
python render_experiment_profiles.py \
  --video primary.mp4 \
  --timestamps primary_timestamps.csv \
  --aux-video auxiliary.mp4 \
  --aux-timestamps auxiliary_timestamps.csv \
  --trace-directory /path/to/constraint_run_logs \
  --output result.mp4 \
  --wait-seconds 0
```

后处理使用 FFmpeg `libx264`、CRF 18。原始视频和轨迹 JSON 不会被修改。

## 拼接展示总视频

默认拼接顺序为 `4, 7, 1, 6, 13, 10, 12, 11`，每两个轨迹之间停顿 2 秒，停顿画面为上一段视频的最后一帧：

```bash
python kuka_experiment_data/video_process/concat_trajectory_videos.py
```

默认输出：

```bash
kuka_experiment_data/learned_from_human/selected_trajectories_dual_profiles.mp4
```

如果要临时调整顺序或停顿时间：

```bash
python kuka_experiment_data/video_process/concat_trajectory_videos.py \
  --order 4 7 1 6 13 10 12 11 \
  --pause-seconds 2
```
