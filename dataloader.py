import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd
import numpy as np
from config import DataConfig

class MELDCombinedDataset(Dataset):

    def __init__(self, config=None, split='train'):
        """
        Combined dataloader that loads:
        - RoBERTa text features from config.roberta_path
        - COMET commonsense features from config.comet_path  
        - Audio and visual features from config.multimodal_path
        
        Args:
            config: DataConfig object with file paths and settings
            split: 'train', 'valid', or 'test'
            classify: 'emotion' or 'sentiment'
        
        label index mapping = {0:neutral, 1:surprise, 2:fear, 3:sadness, 4:joy, 5:disgust, 6:anger}
        """

        self.config = config
    
        # Load RoBERTa text features
        (
            speakers,
            emotion_labels, 
            sentiment_labels,
            self.roberta1,
            self.roberta2, 
            self.roberta3,
            self.roberta4,
            sentences,
            trainIds,
            testIds,
            validIds
        ) = pickle.load(open(config.data_path + config.roberta_path, 'rb'), encoding='latin1')
        
        # Load COMET commonsense features
        (
            self.xIntent,
            self.xAttr, 
            self.xNeed,
            self.xWant,
            self.xEffect,
            self.xReact,
            self.oWant,
            self.oEffect,
            self.oReact
        ) = pickle.load(open(config.data_path + config.comet_path, 'rb'), encoding='latin1')

        # Load multimodal features (audio/visual)
        (
            videoIDs,
            videoSpeakers,
            videoLabels,
            videoSentiments,
            videoText0,
            videoText1,
            videoText2,
            videoText3,
            videoAudio,
            videoVisual,
            videoSentence,
            trainVid,
            testVid,
            _,
        ) = pickle.load(open(config.data_path + config.multimodal_path, 'rb'))

        self.trainIds = trainIds
        self.testIds = testIds
        self.validIds = validIds

        self.emotion_labels = videoLabels
        self.sentiment_labels = videoSentiments

        self.speakers = speakers
        self.sentences = sentences

        self.videoIDs = videoIDs
        self.videoAudio = videoAudio
        self.videoVisual = videoVisual

        if split == 'train':
            self.keys = [x for x in self.trainIds]
        elif split == 'test':
            self.keys = [x for x in self.testIds]
        elif split == 'valid':
            self.keys = [x for x in self.validIds]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return (
            # RoBERTa text features (4 variants)
            torch.FloatTensor(np.array(self.roberta1[vid])),
            torch.FloatTensor(np.array(self.roberta2[vid])),
            torch.FloatTensor(np.array(self.roberta3[vid])),
            torch.FloatTensor(np.array(self.roberta4[vid])),
            # COMET commonsense features (9 types)
            torch.FloatTensor(np.array(self.xIntent[vid])),
            torch.FloatTensor(np.array(self.xAttr[vid])),
            torch.FloatTensor(np.array(self.xNeed[vid])),
            torch.FloatTensor(np.array(self.xWant[vid])),
            torch.FloatTensor(np.array(self.xEffect[vid])),
            torch.FloatTensor(np.array(self.xReact[vid])),
            torch.FloatTensor(np.array(self.oWant[vid])),
            torch.FloatTensor(np.array(self.oEffect[vid])),
            torch.FloatTensor(np.array(self.oReact[vid])),
            # Audio and visual features
            torch.FloatTensor(np.array(self.videoVisual[vid])),
            torch.FloatTensor(np.array(self.videoAudio[vid])),
            # Speaker information
            torch.FloatTensor(np.array(self.speakers[vid])),
            # Mask and labels
            torch.FloatTensor(np.ones(len(np.array(self.emotion_labels[vid])))),
            torch.LongTensor(np.array(self.emotion_labels[vid])),
            torch.LongTensor(np.array(self.sentiment_labels[vid])),
            vid,
        )

    def __len__(self):
        return self.len

    def return_labels(self):
        return_label = []
        for key in self.keys:
            return_label += self.emotion_labels[key]
        return return_label

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        
        # Group items properly for training script expectation
        # RoBERTa features (indices 0-3)
        roberta_features = [pad_sequence(dat[i]) for i in range(4)]
        
        # COMET features (indices 4-12) 
        comet_features = [pad_sequence(dat[i]) for i in range(4, 13)]
        
        # Visual features (index 13)
        visual_features = pad_sequence(dat[13])
        
        # Audio features (index 14)
        audio_features = pad_sequence(dat[14])
        
        # Speakers (index 15)
        speakers = pad_sequence(dat[15])
        
        # qmask - create from speakers (assuming speaker mask is qmask)
        qmask = pad_sequence(dat[15])
        
        # umask (index 16)
        umask = pad_sequence(dat[16])
        
        # Emotion labels (index 17)
        labels_emo = pad_sequence(dat[17], True)
        
        # Sentiment labels (index 18)
        labels_sen = pad_sequence(dat[18], True)
        
        return (roberta_features, comet_features, visual_features, audio_features, 
                speakers, qmask, umask, labels_emo, labels_sen)

# Backward compatibility alias
CombinedDataset = MELDCombinedDataset

# Test print
# if __name__ == "__main__":
#     from config import DataConfig
#     config = DataConfig()
#     dataset = MELDCombinedDataset(config=config)
#     print("Video IDs (first 5):", list(dataset.videoIDs)[:5]) 
#     print(f"Test count: {len(dataset.videoIDs)}")