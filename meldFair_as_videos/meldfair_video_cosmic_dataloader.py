import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import pandas as pd
import numpy as np
import os
import cv2
from PIL import Image
import torchaudio
import torchaudio.transforms as TAT
import torchvision.transforms as TVT

# Audio processing parameters (from MELD-FAIR config)
SR = 16000
N_MFCC = 13
N_MELS = 26


class MELDFAIRVideoDataset_COSMIC_GraphSmile(Dataset):
    """
    MELD-FAIR dataset using full video sequences + COSMIC commonsense features
    for enhanced multimodal emotion recognition with GraphSmile pipeline.
    """

    def __init__(self, meldfair_path, cosmic_roberta_path, cosmic_comet_path, 
                 split='train', max_frames=None, face_size=112):
        """
        Initialize MELD-FAIR Video + COSMIC dataset.
        
        Args:
            meldfair_path: Path to MELD-FAIR base folder
            cosmic_roberta_path: Path to COSMIC's meld_features_roberta.pkl
            cosmic_comet_path: Path to COSMIC's meld_features_comet.pkl
            split: 'train', 'dev', or 'test'
            max_frames: Maximum frames per video (None = use all frames)
            face_size: Size to resize video frames
        """
        self.meldfair_path = meldfair_path
        self.split = split if split != 'dev' else 'dev'  # MELD-FAIR uses 'dev' instead of 'valid'
        self.max_frames = max_frames
        self.face_size = face_size
        
        # Define paths based on MELD-FAIR structure
        self.realigned_csv_path = os.path.join(meldfair_path, 'csvs', f'realigned_{self.split}_sent_emo.csv')
        self.face_bbox_csv_path = os.path.join(meldfair_path, 'csvs', 'MELD_active_speaker_face_bboxes.csv')
        self.audio_folder = os.path.join(meldfair_path, 'MELD', 'realigned', self.split, 'audio', str(SR))
        self.video_folder = os.path.join(meldfair_path, 'MELD', 'realigned', self.split, 'videos')
        
        # Load COSMIC features
        self._load_cosmic_features(cosmic_roberta_path, cosmic_comet_path)
        
        # Load MELD-FAIR data
        self._load_meldfair_data()
        
        # Setup transforms
        self.video_transform = TVT.Compose([
            TVT.Resize((face_size, face_size)),
            TVT.ToTensor(),
            TVT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Audio transforms
        self.mel_spectrogram = TAT.MelSpectrogram(
            sample_rate=SR,
            n_fft=512,
            win_length=int(0.025 * SR),
            hop_length=int(0.01 * SR),
            n_mels=N_MELS
        )
        self.mfcc_transform = TAT.MFCC(
            sample_rate=SR,
            n_mfcc=N_MFCC,
            melkwargs={'n_fft': 512, 'n_mels': N_MELS}
        )
        
    def _load_cosmic_features(self, roberta_path, comet_path):
        """Load COSMIC RoBERTa and COMET features"""
        # Load COSMIC RoBERTa features
        cosmic_roberta_data = pickle.load(open(roberta_path, 'rb'), encoding='latin1')
        (
            self.cosmic_speakers,
            self.cosmic_emotion_labels,
            self.cosmic_sentiment_labels,
            self.cosmic_roberta1,
            self.cosmic_roberta2,
            self.cosmic_roberta3,
            self.cosmic_roberta4,
            self.cosmic_sentences,
            self.cosmic_trainIds,
            self.cosmic_testIds,
            self.cosmic_validIds
        ) = cosmic_roberta_data
        
        # Load COSMIC COMET features (commonsense)
        cosmic_comet_data = pickle.load(open(comet_path, 'rb'), encoding='latin1')
        (
            self.xIntent,    # self intent
            self.xAttr,      # self attributes
            self.xNeed,      # self needs
            self.xWant,      # self wants
            self.xEffect,    # effect on self
            self.xReact,     # self reaction
            self.oWant,      # others want
            self.oEffect,    # effect on others
            self.oReact      # others reaction
        ) = cosmic_comet_data
        
    def _load_meldfair_data(self):
        """Load MELD-FAIR realigned data"""
        # Load realigned CSV
        df_realigned = pd.read_csv(self.realigned_csv_path)
        
        # Load face bounding box info for cropping
        df_faces = pd.read_csv(self.face_bbox_csv_path)
        self.face_bboxes = {}
        for _, row in df_faces.iterrows():
            if row['Split'] == self.split:
                key = (row['Dialogue ID'], row['Utterance ID'], row['Frame Number'])
                self.face_bboxes[key] = {
                    'x_left': row['X Left'],
                    'y_top': row['Y Top'], 
                    'x_right': row['X Right'],
                    'y_bottom': row['Y Bottom']
                }
        
        # Get utterances that have face bounding boxes
        face_utterances = set()
        for (dia_id, utt_id, frame_num) in self.face_bboxes.keys():
            face_utterances.add((dia_id, utt_id))
        
        # Emotion label mapping (same as original MELD)
        emotion_map = {
            'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 
            'joy': 4, 'disgust': 5, 'anger': 6
        }
        
        self.data = []
        
        for _, row in df_realigned.iterrows():
            dia_id = row['Dialogue_ID']
            utt_id = row['Utterance_ID']
            emotion = row['Emotion']
            sentiment = row['Sentiment']
            speaker = row['Speaker']
            utterance = row['Utterance']
            
            # Check if utterance has video data and COSMIC data
            cosmic_key = dia_id  # COSMIC uses dialogue ID as key
            if (dia_id, utt_id) in face_utterances and cosmic_key in self.cosmic_roberta1:
                # Construct file paths
                audio_path = os.path.join(self.audio_folder, f"dia{dia_id}_utt{utt_id}.wav")
                video_path = os.path.join(self.video_folder, f"dia{dia_id}_utt{utt_id}.mp4")
                
                # Check if files exist
                if os.path.exists(audio_path) and os.path.exists(video_path):
                    self.data.append({
                        'dia_id': dia_id,
                        'utt_id': utt_id,
                        'cosmic_key': cosmic_key,
                        'emotion_label': emotion_map.get(emotion, 0),
                        'sentiment_label': 1 if sentiment == 'positive' else (2 if sentiment == 'negative' else 0),
                        'speaker': speaker,
                        'utterance': utterance,
                        'audio_path': audio_path,
                        'video_path': video_path
                    })
        
        print(f"Loaded {len(self.data)} valid video samples for split '{self.split}'")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        dia_id = item['dia_id']
        utt_id = item['utt_id']
        cosmic_key = item['cosmic_key']
        
        # Load audio
        audio_tensor = self._load_audio(item['audio_path'])
        
        # Load video sequence with face cropping
        video_sequence = self._load_video_sequence(item['video_path'], dia_id, utt_id)
        
        # Get COSMIC text features (using dialogue-level features)
        cosmic_dia_idx = self._get_cosmic_dialogue_index(cosmic_key, utt_id)
        
        # COSMIC RoBERTa features (text modalities)
        text_features = self._get_cosmic_text_features(cosmic_key, cosmic_dia_idx)
        
        # COSMIC commonsense features
        commonsense_features = self._get_cosmic_commonsense_features(cosmic_key, cosmic_dia_idx)
        
        return {
            'dia_id': dia_id,
            'utt_id': utt_id,
            'audio': audio_tensor,
            'video_sequence': video_sequence,
            'text_features': text_features,
            'commonsense_features': commonsense_features,
            'emotion_label': item['emotion_label'],
            'sentiment_label': item['sentiment_label'],
            'speaker': item['speaker'],
            'utterance': item['utterance']
        }
    
    def _load_audio(self, audio_path):
        """Load and process audio file"""
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Resample if needed
            if sample_rate != SR:
                resampler = torchaudio.transforms.Resample(sample_rate, SR)
                waveform = resampler(waveform)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Extract MFCC features
            mfcc = self.mfcc_transform(waveform).squeeze(0).transpose(0, 1)  # [time, features]
            
            return mfcc
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            # Return zero tensor as fallback
            return torch.zeros(100, N_MFCC)
    
    def _load_video_sequence(self, video_path, dia_id, utt_id):
        """Load and process full video sequence with face cropping"""
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            frame_idx = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Get face bounding box for this frame
                bbox_key = (dia_id, utt_id, frame_idx)
                if bbox_key in self.face_bboxes:
                    bbox = self.face_bboxes[bbox_key]
                    
                    # Crop face from frame
                    cropped_frame = frame[
                        bbox['y_top']:bbox['y_bottom'],
                        bbox['x_left']:bbox['x_right']
                    ]
                    
                    # Convert BGR to RGB
                    cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                    
                    # Convert to PIL Image and apply transforms
                    pil_frame = Image.fromarray(cropped_frame)
                    frame_tensor = self.video_transform(pil_frame)
                    frames.append(frame_tensor)
                
                frame_idx += 1
                
                # Limit max frames if specified
                if self.max_frames and len(frames) >= self.max_frames:
                    break
            
            cap.release()
            
            # Ensure we have at least one frame
            if len(frames) == 0:
                # Fallback: create a zero tensor
                frames = [torch.zeros(3, self.face_size, self.face_size)]
            
            # Stack frames: [frames, channels, height, width]
            video_tensor = torch.stack(frames)
            
            return video_tensor
            
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            # Return zero tensor as fallback
            return torch.zeros(10, 3, self.face_size, self.face_size)
    
    def _get_cosmic_dialogue_index(self, cosmic_key, utt_id):
        """Get the index within the COSMIC dialogue for this utterance"""
        # COSMIC stores features at dialogue level, we need to map to utterance
        # For simplicity, we'll use utterance ID as index (adjust as needed)
        try:
            max_idx = len(self.cosmic_roberta1[cosmic_key]) - 1
            return min(utt_id, max_idx)
        except KeyError:
            return 0
    
    def _get_cosmic_text_features(self, cosmic_key, cosmic_dia_idx):
        """Get COSMIC RoBERTa text features"""
        try:
            roberta1 = torch.FloatTensor(self.cosmic_roberta1[cosmic_key][cosmic_dia_idx])
            roberta2 = torch.FloatTensor(self.cosmic_roberta2[cosmic_key][cosmic_dia_idx])
            roberta3 = torch.FloatTensor(self.cosmic_roberta3[cosmic_key][cosmic_dia_idx])
            roberta4 = torch.FloatTensor(self.cosmic_roberta4[cosmic_key][cosmic_dia_idx])
            return [roberta1, roberta2, roberta3, roberta4]
        except (KeyError, IndexError) as e:
            print(f"Error getting COSMIC text features for key {cosmic_key}, idx {cosmic_dia_idx}: {e}")
            # Return zero tensors as fallback
            return [torch.zeros(1024) for _ in range(4)]
    
    def _get_cosmic_commonsense_features(self, cosmic_key, cosmic_dia_idx):
        """Get COSMIC COMET commonsense features"""
        try:
            commonsense_feats = [
                torch.FloatTensor(self.xIntent[cosmic_key][cosmic_dia_idx]),
                torch.FloatTensor(self.xAttr[cosmic_key][cosmic_dia_idx]),
                torch.FloatTensor(self.xNeed[cosmic_key][cosmic_dia_idx]),
                torch.FloatTensor(self.xWant[cosmic_key][cosmic_dia_idx]),
                torch.FloatTensor(self.xEffect[cosmic_key][cosmic_dia_idx]),
                torch.FloatTensor(self.xReact[cosmic_key][cosmic_dia_idx]),
                torch.FloatTensor(self.oWant[cosmic_key][cosmic_dia_idx]),
                torch.FloatTensor(self.oEffect[cosmic_key][cosmic_dia_idx]),
                torch.FloatTensor(self.oReact[cosmic_key][cosmic_dia_idx])
            ]
            return commonsense_feats
        except (KeyError, IndexError) as e:
            print(f"Error getting COSMIC commonsense features for key {cosmic_key}, idx {cosmic_dia_idx}: {e}")
            # Return zero tensors as fallback
            return [torch.zeros(768) for _ in range(9)]


def collate_meldfair_video_cosmic(batch):
    """Custom collate function for MELD-FAIR Video + COSMIC data"""
    dia_ids = [item['dia_id'] for item in batch]
    utt_ids = [item['utt_id'] for item in batch]
    
    # Audio features - pad to same length
    audio_features = [item['audio'] for item in batch]
    audio_padded = pad_sequence(audio_features, batch_first=True, padding_value=0)
    
    # Video sequences - pad to same number of frames
    video_sequences = [item['video_sequence'] for item in batch]
    video_padded = pad_sequence(video_sequences, batch_first=True, padding_value=0)
    
    # Text features - stack roberta features
    text_features = []
    for i in range(4):  # 4 RoBERTa layers
        text_layer = torch.stack([item['text_features'][i] for item in batch])
        text_features.append(text_layer)
    
    # Commonsense features - stack COMET features
    commonsense_features = []
    for i in range(9):  # 9 COMET features
        cs_layer = torch.stack([item['commonsense_features'][i] for item in batch])
        commonsense_features.append(cs_layer)
    
    # Labels
    emotion_labels = torch.LongTensor([item['emotion_label'] for item in batch])
    sentiment_labels = torch.LongTensor([item['sentiment_label'] for item in batch])
    
    # Create masks for valid frames
    audio_lengths = [item['audio'].shape[0] for item in batch]
    video_lengths = [item['video_sequence'].shape[0] for item in batch]
    
    max_audio_len = audio_padded.shape[1]
    max_video_len = video_padded.shape[1]
    
    audio_mask = torch.zeros(len(batch), max_audio_len)
    video_mask = torch.zeros(len(batch), max_video_len)
    
    for i, (audio_len, video_len) in enumerate(zip(audio_lengths, video_lengths)):
        audio_mask[i, :audio_len] = 1
        video_mask[i, :video_len] = 1
    
    return {
        'dia_ids': dia_ids,
        'utt_ids': utt_ids,
        'audio': audio_padded,
        'video_sequences': video_padded,
        'text_features': text_features,
        'commonsense_features': commonsense_features,
        'emotion_labels': emotion_labels,
        'sentiment_labels': sentiment_labels,
        'audio_mask': audio_mask,
        'video_mask': video_mask,
        'audio_lengths': audio_lengths,
        'video_lengths': video_lengths
    }


def get_MELDFAIR_VIDEO_COSMIC_loaders(meldfair_path, cosmic_roberta_path, cosmic_comet_path,
                                     batch_size=8, num_workers=0, pin_memory=False, max_frames=None):
    """
    Create data loaders for MELD-FAIR Video + COSMIC training and testing.
    
    Args:
        meldfair_path: Path to MELD-FAIR base folder
        cosmic_roberta_path: Path to COSMIC's meld_features_roberta.pkl
        cosmic_comet_path: Path to COSMIC's meld_features_comet.pkl
        batch_size: Batch size for data loading
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory
        max_frames: Maximum frames per video (None = use all frames)
    
    Returns:
        train_loader, dev_loader, test_loader: DataLoader objects
    """
    
    train_dataset = MELDFAIRVideoDataset_COSMIC_GraphSmile(
        meldfair_path, cosmic_roberta_path, cosmic_comet_path, 
        split='train', max_frames=max_frames
    )
    dev_dataset = MELDFAIRVideoDataset_COSMIC_GraphSmile(
        meldfair_path, cosmic_roberta_path, cosmic_comet_path, 
        split='dev', max_frames=max_frames
    )
    test_dataset = MELDFAIRVideoDataset_COSMIC_GraphSmile(
        meldfair_path, cosmic_roberta_path, cosmic_comet_path, 
        split='test', max_frames=max_frames
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_meldfair_video_cosmic,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )
    
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=batch_size,
        collate_fn=collate_meldfair_video_cosmic,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=collate_meldfair_video_cosmic,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )
    
    return train_loader, dev_loader, test_loader


if __name__ == "__main__":
    # Test the dataloader
    meldfair_path = "../MELD-FAIR"
    cosmic_roberta_path = "../conv-emotion/COSMIC/erc-training/meld/meld_features_roberta.pkl"
    cosmic_comet_path = "../conv-emotion/COSMIC/erc-training/meld/meld_features_comet.pkl"
    
    try:
        train_loader, dev_loader, test_loader = get_MELDFAIR_VIDEO_COSMIC_loaders(
            meldfair_path, cosmic_roberta_path, cosmic_comet_path, batch_size=2, max_frames=50
        )
        
        # Test loading a batch
        for batch in train_loader:
            print(f"Video batch loaded successfully!")
            print(f"Audio shape: {batch['audio'].shape}")
            print(f"Video sequences shape: {batch['video_sequences'].shape}")
            print(f"Text features shapes: {[tf.shape for tf in batch['text_features']]}")
            print(f"Commonsense features shapes: {[cf.shape for cf in batch['commonsense_features']]}")
            print(f"Video lengths: {batch['video_lengths']}")
            print(f"Emotion labels: {batch['emotion_labels']}")
            break
            
    except Exception as e:
        print(f"Error testing video dataloader: {e}")
        import traceback
        traceback.print_exc()