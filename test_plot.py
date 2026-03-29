import torch
from autoemulate.core.compare import AutoEmulate
x = torch.linspace(0, 1, 40).unsqueeze(1)
y = x**2

ae = AutoEmulate(
    x,
    y,
    n_iter=1,
    n_splits=2,
    n_bootstraps=None,
    log_level="error",
)

best = ae.best_result()
ae.plot(best, fname="test_plot_output.png")