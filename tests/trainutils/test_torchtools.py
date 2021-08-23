from vectorrvnn.trainutils import * 

def test_receptive_field_resnet50() :
    model = resnet50()
    rf = cnnReceptiveField(model, (3, 224, 224), (0,))
    assert rf==224

def test_receptive_field() : 
    model = resnet50().conv1
    rf = receptiveField(model, (3, 224, 224), (0, 0, 0))
    diff = list(map(lambda x, y : x - y, model.kernel_size, model.padding))
    assert rf == reduce(lambda x, y : x * y, (model.in_channels, *diff))

def test_cnn_rf() : 
    model = resnet50().conv1
    rf = cnnReceptiveField(model, (3, 224, 224), (0, 10, 10))
    assert rf == model.kernel_size[0]
    
def test_cnn_stride () : 
    model = resnet50().conv1
    stride = cnnEffectiveStride(model, (3, 224, 224), (0, 10, 10))
    assert tuple(stride) == model.stride

