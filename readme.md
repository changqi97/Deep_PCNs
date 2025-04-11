

## Usage:

env: pcax 0.6.2

```bash
python nudging_PC.py VGG13_updateUPCL_CIFAR10.yaml
```

## Note:
1. The weight update method can be changed to the previous method by changing out_key='u' to out_key='h' on line 66.
2. The decay mode can be changed on line 256. For details, refer to the get_sequences() function in util.py.
