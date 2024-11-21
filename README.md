# ADOPT: Modified Adam Can Converge with Any $β_2$ with the Optimal Rate
Official Implementation of "[ADOPT: Modified Adam Can Converge with Any $β_2$ with the Optimal Rate](https://arxiv.org/abs/2411.02853)", which is presented at NeurIPS 2024.

## Update on Nov 22, 2024

Based on feedbacks from many practitioners, we have updated the implementation and the paper to improve the stability of our ADOPT algorithm.
In the original version, ADOPT sometimes gets unstable especially in the early stage of training.
This seems to be because the near-zero division by the second memont estimate occurs when some elements of the parameter gradient is near zero at initialization.
For example, when the some parameters are initialized with zero, which is often-used technique in deep learing, near-zero gradient is observed.
To avoid such near-zero division, we decide to add a clipping operation in the momentum update.
Even when the clipping is applied, the convergence guarantee in theory is maintained by properly scheduling the clipping value (see the updated arXiv paper).
In our implementation, the clipping value is controlled by the argument `clipping_lambda`, which is a callable function that determines the scheduling of the clipping value depending on the number of gradient steps.
By default, the clipping value is set to `step**0.25`, which aligns with the theory to ensure the convergence.
If you want to reproduce the behaivior of the original version, you should set `clipping_lambda = None`.

## Requirements

ADOPT requires PyTorch 2.5.0 or later.

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
