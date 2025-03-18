import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights


class CocktailNet(nn.Module):
    """
    Audio-visual speech separation model.
    Visual path: 3D-ResNet18 encodes lip/face video into a single embedding per clip.
    Audio path: Bi-LSTM processes noisy spectrogram frame-by-frame.
    Fusion: concat visual + audio features, predict a soft mask per time-frequency bin.
    Output mask is applied to noisy magnitude to isolate target speaker.
    """

    def __init__(self, freq_bins=257, lstm_hidden=256, lstm_layers=2, dropout=0.3):
        super().__init__()

        # 3D-ResNet18 pretrained on Kinetics — replace final classifier with identity
        # so forward() returns a 512-dim embedding vector for the whole clip
        self.visual_frontend = r3d_18(weights=R3D_18_Weights.DEFAULT)
        self.visual_frontend.fc = nn.Identity()

        # Bi-LSTM: input is one spectrogram column (257 bins), output is 512 per frame
        self.audio_lstm = nn.LSTM(
            input_size=freq_bins,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # Fusion MLP: 512 (visual) + 512 (audio LSTM) -> 257 (mask per frame)
        self.fusion = nn.Sequential(
            nn.Linear(512 + lstm_hidden * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, freq_bins),
            nn.Sigmoid(),  # mask values in [0, 1]
        )

    def forward(self, video, noisy_mag):
        """
        video     : [B, 3, T_v, H, W]   — RGB frames, T_v=50, H=W=112
        noisy_mag : [B, freq_bins, T_a]  — STFT magnitude, freq_bins=257, T_a=200
        returns   : [B, freq_bins, T_a]  — soft mask, same shape as noisy_mag
        """

        # one 512-dim vector summarising the whole video clip
        visual_feat = self.visual_frontend(video)           # [B, 512]

        # LSTM expects [B, T, features]
        audio_in = noisy_mag.transpose(1, 2)               # [B, T_a, 257]
        lstm_out, _ = self.audio_lstm(audio_in)            # [B, T_a, 512]

        # broadcast visual vector across all audio time steps
        T_a = lstm_out.size(1)
        visual_exp = visual_feat.unsqueeze(1).expand(-1, T_a, -1)  # [B, T_a, 512]

        fused = torch.cat([visual_exp, lstm_out], dim=2)   # [B, T_a, 1024]
        mask = self.fusion(fused)                          # [B, T_a, 257]

        return mask.transpose(1, 2)                        # [B, 257, T_a]
