import os
import pandas as pd
from pydub import AudioSegment

def create_dataset(root_dir):
    """
    Создает Pandas DataFrame с информацией об аудиофайлах.
    """
    data = []
    for profession in os.listdir(root_dir):
        profession_dir = os.path.join(root_dir, profession)
        if not os.path.isdir(profession_dir):
            continue

        for gender in os.listdir(profession_dir):
            gender_dir = os.path.join(profession_dir, gender)
            if not os.path.isdir(gender_dir):
                continue

            for speaker_folder in os.listdir(gender_dir): 
                speaker_dir = os.path.join(gender_dir, speaker_folder)
                if not os.path.isdir(speaker_dir):
                    continue

                for filename in os.listdir(speaker_dir):
                    if filename.endswith(".m4a"):
                        filepath = os.path.join(speaker_dir, filename)
                        
                        speaker_number = speaker_folder.split('_')[-1]  
                        data.append({
                            'filepath': filepath,
                            'profession': profession,
                            'gender': gender,
                            'speaker_number': speaker_number 
                        })

    df = pd.DataFrame(data)
    print("Head of dataframe:", df.head())  
    print("Shape of dataframe:", df.shape) 
    return df


def convert_to_wav(filepath, output_dir):
    """
    Конвертирует m4a в wav.
    """
    try:
        audio = AudioSegment.from_file(filepath, "m4a")
        output_filename = os.path.splitext(os.path.basename(filepath))[0] + ".wav"
        output_path = os.path.join(output_dir, output_filename)
        audio.export(output_path, format="wav")
        return output_path
    except Exception as e:
        print(f"Ошибка конвертации {filepath}: {e}")
        return None

# Основной скрипт 
root_directory = r"C:\Users\user\Desktop\my_dataset\audiofiles"  
output_wav_directory = "wav_dataset" 


os.makedirs(output_wav_directory, exist_ok=True)


df = create_dataset(root_directory)


if 'filepath' in df.columns and not df.empty:
    print("Column 'filepath' exists and DataFrame is not empty. Proceeding with conversion...")
    
    df['wav_filepath'] = df['filepath'].apply(lambda x: convert_to_wav(x, output_wav_directory))
    df = df.dropna(subset=['wav_filepath']) 

    
    df['filepath'] = df['wav_filepath']
    df = df.drop('wav_filepath', axis=1)

    
    df.to_csv("audio_dataset.csv", index=False)

    print("Датасет создан и сохранен в audio_dataset.csv")
else:
    print("Error: 'filepath' column not found in DataFrame or DataFrame is empty. Check your file structure and the create_dataset() function.")



