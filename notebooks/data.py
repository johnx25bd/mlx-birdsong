from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import whisper

class BirdClefDataset(Dataset):
    def __init__(self, df, audio_dir, df_classes, set_name):
        self.df = df
        self.audio_dir = audio_dir
        self.df_classes = df_classes
        self.cls2idx = {cls: i for i, cls in enumerate(df_classes['SPECIES_CODE'].unique())}
        self.target_length = 3000
        self.set = set_name
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = f"./subsample/{self.set}/{row['filename']}"
        species_code = row['primary_label']

        # Prep the classification target
        cls_actual = torch.tensor(self.cls2idx[species_code], dtype=torch.long)


        # Prep the coordinates
        lon_norm = (row['longitude'] + 180.0) / 360.0
        lat_norm = (row['latitude'] + 90.0) / 180.0
        coords_norm = torch.tensor([lon_norm, lat_norm], dtype=torch.float32)


        # Prep the audio
        audio = whisper.load_audio(filename)
        mel = whisper.log_mel_spectrogram(audio)
        
        mel_input = mel[:, :self.target_length] # truncate to max_seq_len — might cut off important information!

        # If the input is shorter than target_length, pad it
        if mel_input.shape[1] < self.target_length:
            pad_length = self.target_length - mel_input.shape[1]
            mel_input = F.pad(mel_input, (0, pad_length))
        mel_input = mel_input.clamp(-1.0, 1.0)
        description = f"id: {self.cls2idx[species_code]}, species: {species_code}, filename: {row['filename']}"
        # Do we need a mask?

        return mel_input, coords_norm, cls_actual, description


# Not necessary on git repo, since we're including the audio files
# import shutil
# import os

# def copy_audio_files(df, dest_dir):    
#     for _, row in df.iterrows():
#         # check if it exists in target directory
#         if os.path.exists(f"{dest_dir}/{row['filename']}"):
#             print(f"File {row['filename']} already exists")
#             continue
#         # copy the file to a new location
#         # make sure the directory exists
#         subdir = row['filename'].split('/')[0]
#         os.makedirs(f"./{dest_dir}/{subdir}", exist_ok=True)
#         shutil.copy(f"../data/birdclef-2024/train_audio/{row['filename']}", f"{dest_dir}/{row['filename']}")

# copy_audio_files(df_train, "./subsample/train")
# copy_audio_files(df_test, "./subsample/test")
