from torch import nn
import torch
from timm import create_model


class CRNNMel(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        proj_channels: int = 128,
        rnn_hidden_size: int = 196,
        rnn_num_layers: int = 2,
        rnn_dropout: float = 0.2,
        rnn_bidirectional: bool = True,
        num_classes: int = 11,
        out_time_steps: int = 40,
    ) -> None:
        super().__init__()

        self.backbone = create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=1,
            features_only=True,
            out_indices=(1, 2),
        )

        ch_stage2, ch_stage3 = self.backbone.feature_info.channels()

        self.proj2 = nn.Conv2d(ch_stage2, proj_channels, kernel_size=1, bias=False)
        self.proj3 = nn.Conv2d(ch_stage3, proj_channels, kernel_size=1, bias=False)

        self.pool2 = nn.AdaptiveAvgPool2d((1, out_time_steps))
        self.pool3 = nn.AdaptiveAvgPool2d((1, out_time_steps))

        rnn_input_size = proj_channels * 2

        self.rnn = nn.GRU(
            input_size=rnn_input_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            dropout=rnn_dropout if rnn_num_layers > 1 else 0.0,
            bidirectional=rnn_bidirectional,
        )

        fc_in = rnn_hidden_size * 2 if rnn_bidirectional else rnn_hidden_size
        self.fc = nn.Linear(fc_in, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat2, feat3 = self.backbone(x)

        feat2 = self.proj2(feat2)
        feat3 = self.proj3(feat3)

        feat2 = self.pool2(feat2)
        feat3 = self.pool3(feat3)

        feat2 = feat2.squeeze(2)
        feat3 = feat3.squeeze(2)

        feat = torch.cat([feat2, feat3], dim=1)
        feat = feat.permute(2, 0, 1)

        rnn_out, _ = self.rnn(feat)
        logits = self.fc(rnn_out)
        return self.log_softmax(logits)

