import torch.nn as nn

from .pointnet2_encoder import PointNet2Encoder
from .point_embed_encoder import PointEmbedEncoder
from .decoder import PointCloudDecoder
from .folding_decoder import FoldingDecoder
from .transformer_decoder import TransformerDecoder
from .diffusion_decoder import DiffusionDecoder


class PointNet2AutoEncoder(nn.Module):
    def __init__(
        self,
        encoder_mode="ssg",
        decoder_mode="mlp",
        latent_dim=128,
        input_channels=0,
        target_num_points=128,
        output_dim=3,
        base_radius=1.0,
        npoint1=32,
        npoint2=16,
        hidden_dim=128,
        k_cov=32,
        k_agg=16,
        use_attention=True,
        decoder_hidden_dim=512,
        folding_grid_dim=1,
        folding_num_folds=2,
    ):
        super().__init__()

        if encoder_mode in {"ssg", "msg"}:
            self.encoder = PointNet2Encoder(
                mode=encoder_mode,
                input_channels=input_channels,
                latent_dim=latent_dim,
                base_radius=base_radius,
                npoint1=npoint1,
                npoint2=npoint2,
            )

        elif encoder_mode in {"point_embed", "cov_embed", "cov_attention"}:
            self.encoder = PointEmbedEncoder(
                input_dim=3 + input_channels,
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                k_cov=k_cov,
                k_agg=k_agg,
                npoints=(64, 32, 16),
                use_attention=use_attention,
            )

        else:
            raise ValueError(
                f"Unknown encoder_mode: {encoder_mode}. "
                "Use 'ssg', 'msg', 'point_embed', 'cov_embed', or 'cov_attention'."
            )

        if decoder_mode == "mlp":
            self.decoder = PointCloudDecoder(
                latent_dim=latent_dim,
                num_points=target_num_points,
                out_dim=output_dim,
            )

        elif decoder_mode == "folding":
            self.decoder = FoldingDecoder(
                latent_dim=latent_dim,
                num_points=target_num_points,
                out_dim=output_dim,
                hidden_dim=512,
                grid_dim=1,
                num_folds=3,
            )

        elif decoder_mode == "transformer":
            self.decoder = TransformerDecoder(
                latent_dim=latent_dim,
                num_points=target_num_points,
                out_dim=output_dim,
            )

        elif decoder_mode == "diffusion":
            self.decoder = DiffusionDecoder(
            latent_dim=latent_dim,
            num_points=target_num_points,
            out_dim=output_dim,
        )

        else:
            raise ValueError(f"Unknown decoder_mode: {decoder_mode}")

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z