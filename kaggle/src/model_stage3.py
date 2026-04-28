from torch import nn
import torch
from timm import create_model

__all__ = ["CRNNMelStage3", "count_parameters", "summarize_parameters"]


def count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    params = module.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def summarize_parameters(module: nn.Module) -> dict[str, int | float]:
    total = count_parameters(module)
    backbone = count_parameters(module.backbone)
    head = total - backbone
    return {
        "total": total,
        "backbone": backbone,
        "head": head,
        "head_share_pct": round(head / total * 100, 2),
    }


class CRNNMelStage3(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        proj_channels: int = 64,
        rnn_hidden_size: int = 128,
        rnn_num_layers: int = 1,
        rnn_dropout: float = 0.1,
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
            out_indices=(2,),
        )

        stage3_channels = self.backbone.feature_info.channels()[-1]

        self.proj3 = nn.Conv2d(stage3_channels, proj_channels, kernel_size=1, bias=False)
        self.pool3 = nn.AdaptiveAvgPool2d((1, out_time_steps))

        self.rnn = nn.GRU(
            input_size=proj_channels,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            dropout=rnn_dropout if rnn_num_layers > 1 else 0.0,
            bidirectional=rnn_bidirectional,
        )

        fc_in = rnn_hidden_size * 2 if rnn_bidirectional else rnn_hidden_size
        self.fc = nn.Linear(fc_in, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat3 = self.backbone(x)[-1]
        feat3 = self.proj3(feat3)
        feat3 = self.pool3(feat3)
        feat3 = feat3.squeeze(2).permute(2, 0, 1)

        rnn_out, _ = self.rnn(feat3)
        logits = self.fc(rnn_out)
        return self.log_softmax(logits)
