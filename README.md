# ADOPT: Modified Adam Can Converge with Any $β_2$ with the Optimal Rate
Official Implementation of "[ADOPT: Modified Adam Can Converge with Any $β_2$ with the Optimal Rate](https://arxiv.org/abs/2411.02853)", which is presented at NeurIPS 2024.

## Requirements

ADOPT requires PyTorch 2.4.0 or later.

## Usage

You can use ADOPT just like any other PyTorch optimizers by copying `adopt.py` to your project.

When you replace the `Adam` optimizer to our `ADOPT`, you should just replace the optimizer as follows:

```python3
from adopt import ADOPT
# optimizer = Adam(model.parameters(), lr=1e-3)
optimizer = ADOPT(model.parameters(), lr=1e-3)
```

When you are using `AdamW` as a default optimizer, you should set `decoupled=True` for our `ADOPT`:

```python3
# optimizer = AdamW(model.parameters(), lr=1e-3)
optimizer = ADOPT(model.parameters(), lr=1e-3, decoupled=True)
```

## Citation
If you use ADOPT in your research, please cite the paper.
```text
@inproceedings{taniguchi2024adopt,
 author={Taniguchi, Shohei and Harada, Keno and Minegishi, Gouki and Oshima, Yuta and Jeong, Seong Cheol and Nagahara, Go and Iiyama, Tomoshi and Suzuki, Masahiro and Iwasawa, Yusuke and Matsuo, Yutaka},
 booktitle = {Advances in Neural Information Processing Systems},
 title = {ADOPT: Modified Adam Can Converge with Any β2 with the Optimal Rate},
 year = {2024}
}
```

## License
[Apache 2.0](./LICENSE)
