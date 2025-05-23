import torch
import torch.nn as nn

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, num_positions, dim):
        """
        Learnable positional encoding.

        Args:
            num_positions (int): The number of positions to encode (e.g., H*W for images, num_points for point clouds).
            dim (int): The dimensionality of the positional encoding.
        """
        super().__init__()
        self.num_positions = num_positions
        self.dim = dim
        # Initialize parameters, these will be learned during training
        self.pe = nn.Parameter(torch.randn(1, num_positions, dim)) # Add batch dim for easier broadcasting initially

    def forward(self, batch_size=None):
        """
        Returns the learned positional encodings.
        If batch_size is provided, repeats the encodings for the batch.
        Otherwise, returns the base (1, num_positions, dim) encoding.

        Args:
            batch_size (int, optional): If provided, repeats the PE for this batch size.
        
        Returns:
            torch.Tensor: Positional encodings of shape (batch_size, num_positions, dim) or (1, num_positions, dim).
        """
        if batch_size:
            return self.pe.repeat(batch_size, 1, 1)
        return self.pe # Returns shape (1, num_positions, dim)

if __name__ == '__main__':
    num_pos = 1024  # Example: 32*32 image
    pe_dim = 256    # Example dimension

    learned_pe_module = LearnedPositionalEncoding(num_positions=num_pos, dim=pe_dim)
    
    # Test without batch size
    pe_single = learned_pe_module()
    print(f"Learned PE shape (single): {pe_single.shape}") # Expected: (1, 1024, 256)

    # Test with batch size
    bs = 4
    pe_batched = learned_pe_module(batch_size=bs)
    print(f"Learned PE shape (batched): {pe_batched.shape}") # Expected: (4, 1024, 256)

    # Test the squeeze(0) as suggested for __getitem__
    # In __getitem__, batch_size is effectively 1 for the single item's PE
    pe_for_item = learned_pe_module(batch_size=1).squeeze(0)
    print(f"Learned PE shape (for one item, squeezed): {pe_for_item.shape}") # Expected: (1024, 256)

    assert pe_single.shape == (1, num_pos, pe_dim)
    assert pe_batched.shape == (bs, num_pos, pe_dim)
    assert pe_for_item.shape == (num_pos, pe_dim)
    print("LearnedPositionalEncoding basic tests passed.")
