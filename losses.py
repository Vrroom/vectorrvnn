def area (bbox) : 
    """
    Compute the area of the bounding box.
    """
    bbox = bbox.view((-1, 4))
    return bbox[:, 2] * bbox[:, 3]

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
    xm1, ym1 = bbox1[:, 0], bbox1[:, 1]
    xM1, yM1 = xm1 + bbox1[:, 2], ym1 + bbox1[:, 3]
    xm2, ym2 = bbox2[:, 0], bbox2[:, 1]
    xM2, yM2 = xm2 + bbox2[:, 2], ym2 + bbox2[:, 3]
    xm, xM = max(xm1, xm2), min(xM1, xM2)
    ym, yM = max(ym1, ym2), min(yM1, yM2)
    intersection = max(xM - xm, 0) * max(yM - ym, 0)
    union = area(bbox1) + area(bbox2)
    return intersection / union
