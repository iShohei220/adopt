# ADOPT: Modified Adam Can Converge with Any $β_2$ with the Optimal Rate
Official Implementation of "ADOPT: Modified Adam Can Converge with Any $β_2$ with the Optimal Rate", which is presented at NeurIPS 2024.

## Requirements

ADOPT requires PyTorch 2.4.0 or later.

## Usage

You can use ADOPT just like any other PyTorch optimizers by copying `adopt.py` to your project.

```python3
from adopt import ADOPT
optimizer = ADOPT(model.parameters(), lr=1e-3)
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

