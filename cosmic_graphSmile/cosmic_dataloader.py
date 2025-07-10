import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import pandas as pd
import numpy as np


class MELDDataset_COSMIC_GraphSmile(Dataset):
    """
    Unified dataset that combines GraphSmile's multimodal features with COSMIC's commonsense features
    for improved emotion recognition on MELD dataset.
    """

    def __init__(self, graphsmile_path, cosmic_roberta_path, cosmic_comet_path, train=True):
        """
        Initialize the dataset with both GraphSmile and COSMIC features.
        
        Args:
            graphsmile_path: Path to GraphSmile's meld_multi_features.pkl
            cosmic_roberta_path: Path to COSMIC's meld_features_roberta.pkl
            cosmic_comet_path: Path to COSMIC's meld_features_comet.pkl
            train: Whether to load training or test data
        """
        
        # Load GraphSmile features
        graphsmile_data = pickle.load(open(graphsmile_path, "rb"))
        (
            self.videoIDs,
            self.videoSpeakers,
            self.videoLabels,
            self.videoSentiments,
            self.videoText0,
            self.videoText1,
            self.videoText2,
            self.videoText3,
            self.videoAudio,
            self.videoVisual,
            self.videoSentence,
            self.trainVid,
            self.testVid,
            _,
        ) = graphsmile_data
        
        # Load COSMIC RoBERTa features
        cosmic_roberta_data = pickle.load(open(cosmic_roberta_path, 'rb'), encoding='latin1')
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
        cosmic_comet_data = pickle.load(open(cosmic_comet_path, 'rb'), encoding='latin1')
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
        
        # Set up data keys
        self.keys = [x for x in (self.trainVid if train else self.testVid)]
        self.len = len(self.keys)
        
        # Labels
        self.labels_emotion = self.videoLabels
        self.labels_sentiment = self.videoSentiments
        
        # Verify data alignment
        self._verify_data_alignment()
    
    def _verify_data_alignment(self):
        """Verify that GraphSmile and COSMIC data are aligned"""
        # Check if the video IDs match between datasets
        sample_keys = list(self.keys)[:5]  # Check first 5 keys
        
        for key in sample_keys:
            if key not in self.cosmic_roberta1:
                print(f"Warning: Key {key} not found in COSMIC RoBERTa features")
            if key not in self.xIntent:
                print(f"Warning: Key {key} not found in COSMIC COMET features")
        
        print(f"Dataset initialized with {self.len} samples")
    
    def __getitem__(self, index):
        vid = self.keys[index]
        
        # GraphSmile features
        text0 = torch.FloatTensor(np.array(self.videoText0[vid]))
        text1 = torch.FloatTensor(np.array(self.videoText1[vid]))
        text2 = torch.FloatTensor(np.array(self.videoText2[vid]))
        text3 = torch.FloatTensor(np.array(self.videoText3[vid]))
        visual = torch.FloatTensor(np.array(self.videoVisual[vid]))
        audio = torch.FloatTensor(np.array(self.videoAudio[vid]))
        speakers = torch.FloatTensor(np.array(self.videoSpeakers[vid]))
        umask = torch.FloatTensor([1] * len(np.array(self.labels_emotion[vid])))
        
        # COSMIC commonsense features
        x_intent = torch.FloatTensor(np.array(self.xIntent[vid]))
        x_attr = torch.FloatTensor(np.array(self.xAttr[vid]))
        x_need = torch.FloatTensor(np.array(self.xNeed[vid]))
        x_want = torch.FloatTensor(np.array(self.xWant[vid]))
        x_effect = torch.FloatTensor(np.array(self.xEffect[vid]))
        x_react = torch.FloatTensor(np.array(self.xReact[vid]))
        o_want = torch.FloatTensor(np.array(self.oWant[vid]))
        o_effect = torch.FloatTensor(np.array(self.oEffect[vid]))
        o_react = torch.FloatTensor(np.array(self.oReact[vid]))
        
        # Labels
        emotion_labels = torch.LongTensor(np.array(self.labels_emotion[vid]))
        sentiment_labels = torch.LongTensor(np.array(self.labels_sentiment[vid]))
        
        return (
            # GraphSmile multimodal features
            text0, text1, text2, text3, visual, audio, speakers, umask,
            # COSMIC commonsense features
            x_intent, x_attr, x_need, x_want, x_effect, x_react, o_want, o_effect, o_react,
            # Labels
            emotion_labels, sentiment_labels,
            # Video ID
            vid,
        )
    
    def __len__(self):
        return self.len
    
    def return_labels(self):
        """Return all emotion labels for the dataset"""
        return_label = []
        for key in self.keys:
            return_label += self.videoLabels[key]
        return return_label
    
    def collate_fn(self, data):
        """Custom collate function for batch processing"""
        dat = pd.DataFrame(data)
        
        # Pad sequences for all features
        return [
            (
                pad_sequence(dat[i])
                if i < 17  # All tensor features need padding
                else pad_sequence(dat[i]) if i < 19  # Labels need padding
                else dat[i].tolist()  # Video IDs don't need padding
            )
            for i in dat
        ]


def get_MELD_COSMIC_loaders(graphsmile_path, cosmic_roberta_path, cosmic_comet_path, 
                           batch_size=32, num_workers=0, pin_memory=False):
    """
    Create data loaders for training and testing with unified COSMIC+GraphSmile features.
    
    Args:
        graphsmile_path: Path to GraphSmile's meld_multi_features.pkl
        cosmic_roberta_path: Path to COSMIC's meld_features_roberta.pkl
        cosmic_comet_path: Path to COSMIC's meld_features_comet.pkl
        batch_size: Batch size for data loading
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory
    
    Returns:
        train_loader, test_loader: DataLoader objects
    """
    
    trainset = MELDDataset_COSMIC_GraphSmile(
        graphsmile_path, cosmic_roberta_path, cosmic_comet_path, train=True
    )
    testset = MELDDataset_COSMIC_GraphSmile(
        graphsmile_path, cosmic_roberta_path, cosmic_comet_path, train=False
    )
    
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        collate_fn=trainset.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        collate_fn=testset.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )
    
    return train_loader, test_loader


if __name__ == "__main__":
    # Test the dataloader
    graphsmile_path = "datasets/meld_multi_features.pkl"
    cosmic_roberta_path = "../conv-emotion/COSMIC/erc-training/meld/meld_features_roberta.pkl"
    cosmic_comet_path = "../conv-emotion/COSMIC/erc-training/meld/meld_features_comet.pkl"
    
    train_loader, test_loader = get_MELD_COSMIC_loaders(
        graphsmile_path, cosmic_roberta_path, cosmic_comet_path, batch_size=4
    )
    
    # Test loading a batch
    for batch in train_loader:
        print(f"Batch loaded with {len(batch)} elements")
        print(f"Text features shape: {batch[0].shape}")
        print(f"Visual features shape: {batch[4].shape}")
        print(f"Audio features shape: {batch[5].shape}")
        print(f"Commonsense features shapes: {[batch[i].shape for i in range(8, 17)]}")
        print(f"Emotion labels shape: {batch[17].shape}")
        break