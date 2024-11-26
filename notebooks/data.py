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
        cls_actual = torch.zeros(len(self.df_classes))
        cls_actual[self.cls2idx[species_code]] = 1.0
        cls_actual = cls_actual

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

        return mel_input, cls_actual, description

