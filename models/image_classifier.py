import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_prob=0.5):
        super(ImageClassifier, self).__init__()

        self.linear_1 = nn.Linear(input_size, hidden_size)
        self.drop = nn.Dropout(dropout_prob)
        self.linear_2 = nn.Linear(hidden_size, num_classes)

    def forward(self, image_features):
        hidden = F.relu(self.linear_1(image_features))
        hidden = self.drop(hidden)
        outputs = self.linear_2(hidden)

        return outputs
