import torch

from .lrcn import LRCN
from .gve import GVE
from .sentence_classifier import SentenceClassifier
from .image_classifier import ImageClassifier
from .image_sentence_classifier import ImageSentenceClassifier


class ModelLoader:
    def __init__(self, args, dataset):
        self.args = args
        self.dataset = dataset

    def lrcn(self):
        # LRCN arguments
        pretrained_model = self.args.pretrained_model
        embedding_size = self.args.embedding_size
        hidden_size = self.args.hidden_size
        vocab_size = len(self.dataset.vocab)

        layers_to_truncate = self.args.layers_to_truncate

        lrcn = LRCN(pretrained_model, embedding_size, hidden_size, vocab_size,
                layers_to_truncate)

        return lrcn

    def gve(self):
        # Make sure dataset returns labels
        self.dataset.set_label_usage(True)
        # GVE arguments
        embedding_size = self.args.embedding_size
        hidden_size = self.args.hidden_size
        vocab_size = len(self.dataset.vocab)
        input_size = self.dataset.input_size
        num_classes = self.dataset.num_classes

        sc = self.sc()
        sc.load_state_dict(torch.load(self.args.sc_ckpt))
        for param in sc.parameters():
            param.requires_grad = False
        sc.eval()
        ic = self.ic()
        ic.load_state_dict(torch.load(self.args.ic_ckpt))
        for param in ic.parameters():
            param.requires_grad = False
        ic.eval()

        gve = GVE(self.args, input_size, embedding_size, hidden_size,
                  vocab_size, ic, sc, num_classes)

        if self.args.weights_ckpt:
            gve.load_state_dict(torch.load(self.args.weights_ckpt))

        return gve



    def sc(self):
        # Make sure dataset returns labels
        self.dataset.set_label_usage(True)
        # Sentence classifier arguments
        embedding_size = self.args.embedding_size
        hidden_size = self.args.hidden_size
        vocab_size = len(self.dataset.vocab)
        num_classes = self.dataset.num_classes

        sc = SentenceClassifier(embedding_size, hidden_size, vocab_size,
                num_classes)

        return sc

    def ic(self):
        # Make sure dataset returns labels
        self.dataset.set_label_usage(True)
        # Image classifier arguments
        input_size = self.dataset.input_size
        num_classes = self.dataset.num_classes

        ic = ImageClassifier(input_size, num_classes)

        return ic

    def isc(self):
        # Make sure dataset returns labels
        self.dataset.set_label_usage(True)
        # Sentence classifier arguments
        input_size = self.dataset.input_size
        embedding_size = self.args.embedding_size
        hidden_size = self.args.hidden_size
        vocab_size = len(self.dataset.vocab)
        num_classes = self.dataset.num_classes

        isc = ImageSentenceClassifier(input_size, embedding_size,
            hidden_size, vocab_size, num_classes)

        return isc
