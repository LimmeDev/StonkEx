import torch

from stonkex.models.transformer import TemporalTransformer, TransformerConfig


def test_transformer_forward_shape():
    config = TransformerConfig(input_dim=5, d_model=32, nhead=4, num_layers=2, dim_feedforward=64, horizon=6)
    model = TemporalTransformer(config)
    batch = torch.randn(3, 20, 5)
    output = model(batch)
    assert output.shape == (3, 6)
