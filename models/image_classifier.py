import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageClassifier(nn.Module):
    def __init__(self, input_size, num_classes, dropout_prob=0.5):
        super(ImageClassifier, self).__init__()

        self.drop = nn.Dropout(dropout_prob)
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, image_features):
        image_features = self.drop(image_features)
        outputs = self.linear(image_features)

        return outputs
