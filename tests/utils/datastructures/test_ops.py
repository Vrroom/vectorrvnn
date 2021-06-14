from vectorrvnn.utils import *
import torch 

def test_deep_keys () : 
    d = dict(
        key1=dict(
            key2=1,
            key3=2,
        ),
        key4='string',
        key5=dict(
            key6=3,
            key7=123,
        )
    )
    keys = list(deepKeys(d))
    lengths = list(map(len, keys))
    assert(len(keys) == 7)
    assert(max(lengths) == 2)

def test_aggregate_dict () : 
    listOfDicts = []
    for i in range(10) : 
        listOfDicts.append(dict(
            ref=dict(
                crop=torch.randn(3, 64, 64),
                whole=torch.randn(3, 64, 64)
            ),
            plus=dict(
                crop=torch.randn(3, 64, 64),
                whole=torch.randn(3, 64, 64)
            ),
            minus=dict(
                crop=torch.randn(3, 64, 64),
                whole=torch.randn(3, 64, 64)
            ),
            im=torch.randn(4, 100, 100)
        ))
    result = aggregateDict(listOfDicts, torch.stack)
    assert(result['im'].shape == torch.Size([10, 4, 100, 100]))
    assert(result['ref']['crop'].shape == torch.Size([10, 3, 64, 64]))

