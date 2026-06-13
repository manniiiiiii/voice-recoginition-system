import json
import wave
import threading
import pyaudio
import numpy as np
import os
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
from vosk import Model, KaldiRecognizer
import noisereduce as nr
import librosa
import soundfile as sf

class VoiceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("实时语音识别系统")
        
        # 获取屏幕尺寸，设置合适的窗口大小
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = min(1200, screen_width - 100)
        window_height = min(850, screen_height - 100)
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.configure(bg='#f0f0f0')
        
        # 状态变量
        self.is_listening = False
        self.stream = None
        self.p = None
        self.model = None
        self.recognizer = None
        self.noise_profile = None
        self.recognition_thread = None
        
        # 降噪状态记录（用于对比显示）
        self.current_denoise_state = True
        
        # 设置字体
        self.title_font = ('Microsoft YaHei', 20, 'bold')
        self.subtitle_font = ('Microsoft YaHei', 11)
        self.button_font = ('Microsoft YaHei', 12)
        self.text_font = ('Microsoft YaHei', 12)
        self.small_font = ('Microsoft YaHei', 9)
        
        # 先创建界面
        self.create_widgets()
        
        # 然后加载模型
        self.init_model()
        
    def init_model(self):
        """加载Vosk模型"""
        model_path = "vosk-model-cn-0.22"
        if not os.path.exists(model_path):
            model_path = "vosk-model-small-cn-0.22"
        
        if os.path.exists(model_path):
            try:
                self.model = Model(model_path)
                self.status_label.config(text="[成功] 模型已加载: " + model_path, foreground="green")
                self.add_result("[成功] 模型加载成功: " + model_path)
            except Exception as e:
                self.status_label.config(text=f"[错误] 模型加载失败: {e}", foreground="red")
                self.add_result(f"[错误] 模型加载失败: {e}")
        else:
            self.status_label.config(text="[错误] 未找到模型文件", foreground="red")
            self.add_result("[错误] 未找到模型文件，请确保 vosk-model-cn-0.22 或 vosk-model-small-cn-0.22 存在")
    
    def create_widgets(self):
        """创建界面组件"""
        # 标题栏
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, text="实时语音识别系统", 
                                font=self.title_font, 
                                bg='#2c3e50', fg='white')
        title_label.pack(pady=(20, 5))
        
        subtitle_label = tk.Label(title_frame, text="噪声环境下的鲁棒性语音识别", 
                                   font=self.subtitle_font, 
                                   bg='#2c3e50', fg='#ecf0f1')
        subtitle_label.pack()
        
        # 主控制面板
        control_frame = tk.Frame(self.root, bg='#ecf0f1', padx=20, pady=15)
        control_frame.pack(fill=tk.X)
        
        # 按钮区域
        btn_frame = tk.Frame(control_frame, bg='#ecf0f1')
        btn_frame.pack()
        
        self.listen_btn = tk.Button(btn_frame, text="开始识别", 
                                     command=self.toggle_listening,
                                     font=self.button_font,
                                     bg='#27ae60', fg='white',
                                     padx=30, pady=10,
                                     cursor="hand2",
                                     relief=tk.RAISED,
                                     bd=2)
        self.listen_btn.pack(side=tk.LEFT, padx=10)
        
        self.file_btn = tk.Button(btn_frame, text="识别音频文件", 
                                   command=self.recognize_file,
                                   font=self.button_font,
                                   bg='#3498db', fg='white',
                                   padx=20, pady=10,
                                   cursor="hand2",
                                   relief=tk.RAISED,
                                   bd=2)
        self.file_btn.pack(side=tk.LEFT, padx=10)
        
        self.clear_btn = tk.Button(btn_frame, text="清空结果", 
                                    command=self.clear_results,
                                    font=self.button_font,
                                    bg='#e74c3c', fg='white',
                                    padx=20, pady=10,
                                    cursor="hand2",
                                    relief=tk.RAISED,
                                    bd=2)
        self.clear_btn.pack(side=tk.LEFT, padx=10)
        
        # 降噪开关
        self.denoise_var = tk.BooleanVar(value=True)
        denoise_check = tk.Checkbutton(btn_frame, text="降噪模式", 
                                        variable=self.denoise_var,
                                        font=self.button_font,
                                        bg='#ecf0f1',
                                        command=self.toggle_denoise)
        denoise_check.pack(side=tk.LEFT, padx=20)
        
        # 状态栏
        status_frame = tk.Frame(control_frame, bg='#ecf0f1', pady=10)
        status_frame.pack(fill=tk.X, pady=5)
        
        self.status_label = tk.Label(status_frame, text="[等待] 正在初始化...", 
                                      font=self.small_font,
                                      bg='#ecf0f1')
        self.status_label.pack(side=tk.LEFT)
        
        # 实时结果展示区域
        result_frame = tk.LabelFrame(self.root, text="识别结果", 
                                      font=self.button_font,
                                      bg='white', padx=10, pady=10)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.result_text = scrolledtext.ScrolledText(result_frame, 
                                                       font=self.text_font,
                                                       height=6,
                                                       wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        
        # 降噪效果对比区域
        compare_frame = tk.LabelFrame(self.root, text="降噪效果对比", 
                                       font=self.button_font,
                                       bg='white', padx=10, pady=10)
        compare_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # 配置网格布局
        compare_frame.columnconfigure(0, weight=1)
        compare_frame.columnconfigure(1, weight=1)
        compare_frame.rowconfigure(0, weight=1)
        
        # 左侧：无降噪
        frame_left = tk.Frame(compare_frame, bg='#ffe6e6')
        frame_left.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        
        tk.Label(frame_left, text="[无降噪模式]", font=self.button_font,
                 bg='#ffe6e6', fg='#c0392b').pack(pady=5)
        self.no_denoise_text = scrolledtext.ScrolledText(frame_left, 
                                                           font=self.text_font,
                                                           height=8,
                                                           wrap=tk.WORD,
                                                           bg='#fff5f5')
        self.no_denoise_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 右侧：有降噪
        frame_right = tk.Frame(compare_frame, bg='#e6ffe6')
        frame_right.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)
        
        tk.Label(frame_right, text="[启用降噪模式]", font=self.button_font,
                 bg='#e6ffe6', fg='#27ae60').pack(pady=5)
        self.with_denoise_text = scrolledtext.ScrolledText(frame_right, 
                                                             font=self.text_font,
                                                             height=8,
                                                             wrap=tk.WORD,
                                                             bg='#f0fff0')
        self.with_denoise_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 初始化对比区域的提示文字
        self.no_denoise_text.insert(tk.END, "[等待] 开始识别后将在此显示无降噪结果...\n")
        self.with_denoise_text.insert(tk.END, "[等待] 开始识别后将在此显示有降噪结果...\n")
        
        # 提示信息
        tip_frame = tk.Frame(self.root, bg='#f0f0f0')
        tip_frame.pack(fill=tk.X, padx=20, pady=(0, 10))
        
        tip_label = tk.Label(tip_frame, 
                             text="提示：点击「开始实时识别」后说话，系统会实时显示识别结果。启用降噪可提高噪声环境下的识别准确率。",
                             font=self.small_font,
                             bg='#f0f0f0', fg='#7f8c8d')
        tip_label.pack()
        
        # 初始化提示
        self.result_text.insert(tk.END, "[系统] 程序已启动，等待操作...\n")
    
    def toggle_denoise(self):
        """切换降噪状态"""
        status = "启用" if self.denoise_var.get() else "禁用"
        self.current_denoise_state = self.denoise_var.get()
        self.add_result(f"[降噪] 已{status}")
        
        # 更新对比区域标题
        if self.denoise_var.get():
            self.add_result("[提示] 降噪已启用", "with_denoise")
        else:
            self.add_result("[提示] 降噪已禁用", "no_denoise")
    
    def add_result(self, text, target="main"):
        """添加识别结果"""
        if target == "main":
            self.result_text.insert(tk.END, text + "\n")
            self.result_text.see(tk.END)
        elif target == "no_denoise":
            # 清空等待提示
            if self.no_denoise_text.get(1.0, tk.END).strip() == "[等待] 开始识别后将在此显示无降噪结果...":
                self.no_denoise_text.delete(1.0, tk.END)
            self.no_denoise_text.insert(tk.END, text + "\n")
            self.no_denoise_text.see(tk.END)
        elif target == "with_denoise":
            # 清空等待提示
            if self.with_denoise_text.get(1.0, tk.END).strip() == "[等待] 开始识别后将在此显示有降噪结果...":
                self.with_denoise_text.delete(1.0, tk.END)
            self.with_denoise_text.insert(tk.END, text + "\n")
            self.with_denoise_text.see(tk.END)
    
    def clear_results(self):
        """清空所有结果"""
        self.result_text.delete(1.0, tk.END)
        self.no_denoise_text.delete(1.0, tk.END)
        self.with_denoise_text.delete(1.0, tk.END)
        
        self.result_text.insert(tk.END, "[系统] 已清空所有结果\n")
        self.no_denoise_text.insert(tk.END, "[等待] 开始识别后将在此显示无降噪结果...\n")
        self.with_denoise_text.insert(tk.END, "[等待] 开始识别后将在此显示有降噪结果...\n")
    
    def toggle_listening(self):
        """切换实时识别状态"""
        if not self.is_listening:
            self.start_listening()
        else:
            self.stop_listening()
    
    def start_listening(self):
        """开始实时识别"""
        if not self.model:
            self.add_result("[错误] 模型未加载，请检查模型文件")
            return
        
        try:
            # 清空之前的对比区域
            self.no_denoise_text.delete(1.0, tk.END)
            self.with_denoise_text.delete(1.0, tk.END)
            self.no_denoise_text.insert(tk.END, "[实时] 识别中...\n")
            self.with_denoise_text.insert(tk.END, "[实时] 识别中...\n")
            
            self.p = pyaudio.PyAudio()
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=4096
            )
            
            self.recognizer = KaldiRecognizer(self.model, 16000)
            self.is_listening = True
            self.listen_btn.config(text="停止识别", bg='#e74c3c')
            self.status_label.config(text="[状态] 正在监听...", foreground="green")
            self.add_result("[系统] 开始实时识别，请说话...")
            
            # 初始化噪声样本
            self.noise_profile = None
            
            # 启动识别线程
            self.recognition_thread = threading.Thread(target=self.recognition_loop)
            self.recognition_thread.daemon = True
            self.recognition_thread.start()
            
        except Exception as e:
            self.add_result(f"[错误] 启动失败: {e}")
    
    def stop_listening(self):
        """停止实时识别"""
        self.is_listening = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.p:
            self.p.terminate()
        
        self.listen_btn.config(text="开始实时识别", bg='#27ae60')
        self.status_label.config(text="[状态] 已停止", foreground="gray")
        self.add_result("\n[系统] 识别已停止")
    
    def recognition_loop(self):
        """实时识别循环"""
        frame_count = 0
        while self.is_listening:
            try:
                data = self.stream.read(4096, exception_on_overflow=False)
                frame_count += 1
                
                if self.denoise_var.get():
                    # 启用降噪
                    audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    # 收集噪声样本（前10帧）
                    if self.noise_profile is None and frame_count < 10:
                        if frame_count == 1:
                            self.noise_profile = []
                        self.noise_profile.extend(audio_data)
                        if frame_count == 9:
                            self.noise_profile = np.array(self.noise_profile)
                    elif self.noise_profile is not None and len(self.noise_profile) > 0:
                        try:
                            noise_len = min(len(self.noise_profile), len(audio_data))
                            denoised = nr.reduce_noise(y=audio_data, sr=16000, 
                                                        y_noise=self.noise_profile[:noise_len],
                                                        prop_decrease=0.7)
                            denoised = np.clip(denoised, -1, 1)
                            denoised = (denoised * 32768).astype(np.int16)
                            data = denoised.tobytes()
                        except Exception as e:
                            pass
                
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get("text", "")
                    if text:
                        # 显示在主区域
                        self.root.after(0, self.add_result, f"[识别] {text}")
                        # 根据降噪状态显示在对比区域
                        if self.denoise_var.get():
                            self.root.after(0, self.add_result, f"[识别] {text}", "with_denoise")
                        else:
                            self.root.after(0, self.add_result, f"[识别] {text}", "no_denoise")
                        
            except Exception as e:
                if self.is_listening:
                    self.root.after(0, self.add_result, f"[警告] 识别错误: {e}")
                break
    
    def recognize_file(self):
        """识别音频文件"""
        file_path = filedialog.askopenfilename(
            title="选择音频文件",
            filetypes=[("音频文件", "*.wav *.mp3 *.m4a"), ("所有文件", "*.*")]
        )
        
        if not file_path:
            return
        
        self.add_result(f"\n[文件] 正在识别: {os.path.basename(file_path)}")
        
        # 清空对比区域
        self.no_denoise_text.delete(1.0, tk.END)
        self.with_denoise_text.delete(1.0, tk.END)
        self.no_denoise_text.insert(tk.END, "[处理] 正在识别无降噪模式...\n")
        self.with_denoise_text.insert(tk.END, "[处理] 正在识别有降噪模式...\n")
        
        # 在新线程中处理
        thread = threading.Thread(target=self.process_audio_file, args=(file_path,))
        thread.daemon = True
        thread.start()
    
    def process_audio_file(self, file_path):
        """处理音频文件"""
        try:
            # 加载音频
            y, sr = librosa.load(file_path, sr=16000, mono=True)
            
            # 无降噪识别
            text_no_denoise = self.recognize_without_denoise(y, sr)
            self.root.after(0, lambda: self.clear_and_add("no_denoise", f"[无降噪] {text_no_denoise}"))
            
            # 有降噪识别
            text_with_denoise = self.recognize_with_denoise(y, sr)
            self.root.after(0, lambda: self.clear_and_add("with_denoise", f"[有降噪] {text_with_denoise}"))
            
            self.root.after(0, lambda: self.add_result(f"[完成] 文件识别完成"))
            
        except Exception as e:
            self.root.after(0, lambda: self.add_result(f"[错误] 识别失败: {e}"))
    
    def clear_and_add(self, target, text):
        """清空并添加文本到指定区域"""
        if target == "no_denoise":
            self.no_denoise_text.delete(1.0, tk.END)
            self.no_denoise_text.insert(tk.END, text + "\n")
            self.no_denoise_text.see(tk.END)
        elif target == "with_denoise":
            self.with_denoise_text.delete(1.0, tk.END)
            self.with_denoise_text.insert(tk.END, text + "\n")
            self.with_denoise_text.see(tk.END)
    
    def recognize_without_denoise(self, y, sr):
        """无降噪识别"""
        temp_file = "temp_no_denoise.wav"
        try:
            sf.write(temp_file, y, sr, subtype='PCM_16')
            
            wf = wave.open(temp_file, "rb")
            rec = KaldiRecognizer(self.model, sr)
            
            text_parts = []
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    res = json.loads(rec.Result())
                    if text := res.get("text", ""):
                        text_parts.append(text)
            
            final = json.loads(rec.FinalResult())
            if final_text := final.get("text", ""):
                if final_text not in text_parts:
                    text_parts.append(final_text)
            
            wf.close()
            return " ".join(text_parts) if text_parts else "[未识别到内容]"
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def recognize_with_denoise(self, y, sr):
        """有降噪识别（带降噪处理）"""
        temp_file = "temp_with_denoise.wav"
        try:
            # 降噪处理
            noise_len = min(int(0.3 * sr), len(y))
            noise_sample = y[:noise_len]
            y_denoised = nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample, prop_decrease=0.7)
            
            sf.write(temp_file, y_denoised, sr, subtype='PCM_16')
            
            wf = wave.open(temp_file, "rb")
            rec = KaldiRecognizer(self.model, sr)
            
            text_parts = []
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    res = json.loads(rec.Result())
                    if text := res.get("text", ""):
                        text_parts.append(text)
            
            final = json.loads(rec.FinalResult())
            if final_text := final.get("text", ""):
                if final_text not in text_parts:
                    text_parts.append(final_text)
            
            wf.close()
            return " ".join(text_parts) if text_parts else "[未识别到内容]"
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

# 运行应用
if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceRecognitionApp(root)
    root.mainloop()