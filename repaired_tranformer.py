# Accurate-WinCLIP-pytorch-main/src/open_clip/transformer.py 466行-497行
for mask_scale in mask:
    x_select = []
    mask_scale = mask_scale.T
    mask_num, L = mask_scale.shape
    # Ensure class_index doesn't cause out-of-bounds access
    class_index = torch.zeros((mask_scale.shape[0], 1), dtype=torch.int64).to(mask_scale.device)
    mask_scale = torch.cat((class_index, mask_scale.long()), dim=1)

    for i in mask_scale:
        # Add bounds checking
        valid_indices = i[(i >= 0) & (i < x.shape[1])]
        if len(valid_indices) > 0:
            selected = torch.index_select(x, 1, valid_indices)
            x_select.append(selected)
    
    if x_select:  # Only proceed if we have valid selections
        # Stack tensors instead of concatenating to avoid size mismatch issues
        # First, we need to make sure all tensors have the same shape
        # Find the maximum size in dimension 1
        if len(x_select) > 1:
            max_dim1 = max([t.shape[1] for t in x_select])
            min_dim1 = min([t.shape[1] for t in x_select])
            
            # If there's variation in sizes, we'll stack only the common part
            if max_dim1 != min_dim1:
                # Use only the first min_dim1 elements from each tensor
                trimmed_x_select = [t[:, :min_dim1, :] for t in x_select]
                x_scale = torch.cat(trimmed_x_select)
            else:
                x_scale = torch.cat(x_select)
        else:
            x_scale = torch.cat(x_select) if len(x_select) > 1 else x_select[0]
