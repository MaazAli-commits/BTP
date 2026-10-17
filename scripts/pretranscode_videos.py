import os
import subprocess
import imageio_ffmpeg
import json

ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
cache_dir = "static/video_cache"
os.makedirs(cache_dir, exist_ok=True)

with open("demo_data.json", "r") as f:
    demo_data = json.load(f)

# Collect all preset videos and first 50 sample gallery videos
videos_to_convert = []
for p in demo_data['presets']:
    videos_to_convert.append((p['dataset'], p['target_video']))

for item in demo_data['sample_gallery'][:60]:
    videos_to_convert.append((item['dataset'], item['video_file']))

# Deduplicate
videos_to_convert = list(set(videos_to_convert))
print(f"Pre-converting {len(videos_to_convert)} key videos to browser-native H.264 MP4...")

converted = 0
for ds, filename in videos_to_convert:
    src = f"data/TVPReID/TVPReid-{ds}/videos/{filename}"
    dst = os.path.join(cache_dir, f"{ds}_{filename}")
    if os.path.exists(dst):
        continue
    if not os.path.exists(src):
        continue
    cmd = [
        ffmpeg, '-y', '-i', src,
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        dst
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    converted += 1

print(f"Successfully converted {converted} videos to browser-compatible H.264 format in {cache_dir}!")
