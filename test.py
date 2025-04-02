# import numpy as np
# import wespeaker
# from pathlib import Path
# import argparse
# import os
# import librosa

# def get_audio_path(audio_dir):
#     """
#     Recursively finds all audio files in the specified directory.
#     """
#     audio_dir = Path(audio_dir)
#     audio_files = list(audio_dir.glob('**/*.wav')) #+ list(
#         #audio_dir.glob('**/*.mp3')) + list(
#         #audio_dir.glob('**/*.m4a'))

#     return audio_files


# def extract_embeddings(audio_files, device, pretraindir):  # Исправлено название аргумента
#     """
#     Extracts embeddings from audio files using the WeSpeaker model
#     """
#     model = wespeaker.load_model_local(pretraindir)  # Исправлено название аргумента
#     model.set_device(device)

#     embeddings = []
#     labels = []  # Добавляем список для меток

#     for file_path in audio_files:
#         try:
#             label = str(file_path).split(os.sep)[-2]  # Извлекаем метку из пути
#             embedding = model.extract_embedding(file_path)
#             embedding = embedding.cpu().numpy()
#             embeddings.append(embedding)
#             labels.append(label)  # Добавляем метку в список

#         except Exception as e:
#             print(f"Error processing {file_path}: {e}")
#             continue

#     return np.array(embeddings), np.array(labels)  # Возвращаем массивы NumPy

# if __name__ == "__main__":
#     # Замените эти пути на свои
#     train_dir = r"C:\Users\user\Desktop\my_dataset\profession_classificator_convert\train_audio"
#     test_dir = r"C:\Users\user\Desktop\my_dataset\profession_classificator_convert\test_audio"
#     pretraindir = r"C:\Users\user\Desktop\my_dataset\pretraindir"
#     device = "cpu"  # или "cuda", если у вас есть GPU

#     train_audio_files = get_audio_path(train_dir)
#     test_audio_files = get_audio_path(test_dir)
#     all_audio_files = train_audio_files + test_audio_files
#     embeddings, labels = extract_embeddings(all_audio_files, device, pretraindir)

#     print(f"Shape of embeddings: {embeddings.shape}")
#     print(f"Shape of labels: {labels.shape}")
#     print(f"Example embedding: {embeddings[0]}")
#     print(f"Example label: {labels[0]}")

#     # Сохраняем в файлы .npy
#     np.save("embeddings.npy", embeddings)
#     np.save("labels.npy", labels)

#     print("Embeddings and labels saved to embeddings.npy and labels.npy")

# import os

# def counter(path):
#     wav = 0
#     mp3 = 0
#     m4a = 0
#     else_count = 0

#     for root, _, files in os.walk(path):  # os.walk обходит директорию рекурсивно
#         for file in files:
#             if file.endswith(".wav"):
#                 wav += 1
#             elif file.endswith(".mp3"):
#                 mp3 += 1
#             elif file.endswith(".m4a"):
#                 m4a += 1
#             else:
#                 else_count += 1

#     print(f" wav {wav}, mp3 {mp3}, m4a {m4a}, else {else_count}")
# counter (r"C:\Users\user\Desktop\my_dataset\profession_classificator_convert\train_audio")
# counter (r"C:\Users\user\Desktop\my_dataset\profession_classificator_convert\test_audio")

import numpy as np
embeddings = np.load(r"C:\Users\user\Desktop\my_dataset\embs\embeddings.npy")
print(f"Embeddings shape: {embeddings.shape}")
print(f"Embeddings dtype: {embeddings.dtype}")
print(embeddings[:5])  # Вывод первых 5 строк