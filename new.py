# import os
# import subprocess

# def convert_audio(input_file, output_file):
#     """Converts an audio file to WAV format (16 kHz, mono, pcm_s16le)."""
#     command = [
#         'ffmpeg',
#         '-i', input_file,
#         '-acodec', 'pcm_s16le',
#         '-ac', '1',
#         '-ar', '16000',
#         output_file
#     ]
#     try:
#         result = subprocess.run(command, check=True, capture_output=True, text=True)
#         print(f"Successfully converted {input_file} to {output_file}")
#         print(f"FFmpeg output: {result.stdout}")  # Выводим stdout
#     except subprocess.CalledProcessError as e:
#         print(f"Error converting {input_file}:")
#         print(f"Return code: {e.returncode}")  # Выводим код возврата
#         print(f"Stdout: {e.stdout}")  # Выводим stdout
#         print(f"Stderr: {e.stderr}")  # Выводим stderr

# def process_directory(input_dir, output_dir):
#     """Converts all audio files in a directory to WAV format."""
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     for root, _, files in os.walk(input_dir):
#         for file in files:
#             if file.endswith(('.mp3', '.m4a', '.wav')):  # Add other formats if needed
#                 input_file = os.path.join(root, file)
#                 relative_path = os.path.relpath(input_file, input_dir)
#                 output_file = os.path.join(output_dir, relative_path.replace(os.path.splitext(relative_path)[1], '.wav'))

#                 output_dir_for_file = os.path.dirname(output_file)
#                 if not os.path.exists(output_dir_for_file):
#                     os.makedirs(output_dir_for_file)

#                 convert_audio(input_file, output_file)

# if __name__ == "__main__":
#     input_directory = r"C:\Users\user\Desktop\my_dataset\profession_classificator_converted"  # Replace with your input directory
#     output_directory = "C:/Users/user/Desktop/my_dataset/profession_classificator_convert"  # Replace with your output directory
#     process_directory(input_directory, output_directory)

import numpy as np

loaded_data = np.load("C:\\Users\\user\\Desktop\\my_dataset\\embs\\numpy_embs.npy", allow_pickle=True)
print(type(loaded_data))
print(loaded_data.shape)
print(loaded_data[0]) #Посмотрим на первый элемент
print(loaded_data[1]) #Посмотрим на второй элемент