import os
import scipy.io as sio
import soundfile as sf
import numpy as np

def convert_mat_to_wav(mat_dir, output_dir="noise_wav"):
    """
    将 NOISEX-92 的 .mat 文件转换为 .wav 文件
    
    Args:
        mat_dir: 存放 .mat 文件的文件夹路径
        output_dir: 输出 .wav 文件的文件夹路径
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取所有 .mat 文件
    mat_files = [f for f in os.listdir(mat_dir) if f.endswith('.mat')]
    
    print(f"找到 {len(mat_files)} 个 .mat 文件")
    print("=" * 50)
    
    for mat_file in mat_files:
        mat_path = os.path.join(mat_dir, mat_file)
        
        try:
            # 读取 .mat 文件
            mat_data = sio.loadmat(mat_path)
            
            # NOISEX-92 的噪声通常存储在变量 'x' 或 'data' 中
            # 尝试常见的变量名
            noise_data = None
            for var_name in ['x', 'data', 'noise', 'X', 'Data']:
                if var_name in mat_data:
                    noise_data = mat_data[var_name]
                    break
            
            # 如果没找到，获取第一个数值数组
            if noise_data is None:
                for key in mat_data:
                    if isinstance(mat_data[key], np.ndarray) and len(mat_data[key]) > 1000:
                        noise_data = mat_data[key]
                        break
            
            if noise_data is not None:
                # 转换为 1D 数组
                noise_data = noise_data.flatten()
                
                # 归一化到 [-1, 1] 范围
                max_val = np.max(np.abs(noise_data))
                if max_val > 0:
                    noise_data = noise_data / max_val
                else:
                    noise_data = noise_data / 32768.0  # 假设是 int16 范围
                
                # 确保采样率为 16000 Hz（NOISEX-92 通常是 16000 或 20000）
                # 输出文件名
                wav_name = mat_file.replace('.mat', '.wav')
                wav_path = os.path.join(output_dir, wav_name)
                
                # 保存为 wav（16kHz）
                sf.write(wav_path, noise_data, 16000, subtype='PCM_16')
                print(f"✅ 转换成功: {mat_file} -> {wav_name}")
            else:
                print(f"❌ 无法读取: {mat_file}")
                
        except Exception as e:
            print(f"❌ 转换失败: {mat_file}, 错误: {e}")
    
    print("=" * 50)
    print(f"转换完成！输出目录: {output_dir}")

# 运行转换
if __name__ == "__main__":
    # 请修改为你的 .mat 文件所在路径
    mat_dir = "NOISEX-92_mat"  # 你的 .mat 文件夹
    
    if not os.path.exists(mat_dir):
        print(f"❌ 找不到文件夹: {mat_dir}")
        print("请修改 mat_dir 变量为正确的路径")
    else:
        convert_mat_to_wav(mat_dir, "noise_wav")