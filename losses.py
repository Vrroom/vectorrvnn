def area (bbox) : 
    """
    Compute the area of the bounding box.
    """
    bbox = bbox.view((-1, 4))
    return (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])

def iou (bbox1, bbox2) : 
    """
    Compute the intersection over union for two bounding
    boxes.

    Assume that there are N querying boxes and N target
    boxes. Hence the inputs have shape N x 4. The first
    two columns contain the coordinate of the top-left 
    corner and the last two contain the coordinates of
    the bottom-right corner.

    Parameters
    ----------
    bbox1 : torch.tensor
        Query bounding box.
    bbox2 : torch.tensor
        Target bounding box.
    """
    bbox1, bbox2 = bbox1.view((-1, 4)), bbox2.view((-1, 4))
    xMin, yMin = max(bbox1[:, 0], bbox2[:, 0]), max(bbox1[:, 1], bbox2[:, 1])
    xMax, yMax = min(bbox1[:, 2], bbox2[:, 2]), min(bbox1[:, 3], bbox2[:, 3])
    intersection = (xMax - xMin) * (yMax - yMin)
    union = area(bbox1) + area(bbox2)
    return intersection / union
