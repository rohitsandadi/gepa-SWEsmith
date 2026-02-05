# GEPA + SWE-smith Integration

Applying GEPA prompt optimization to software engineering tasks using SWE-smith's synthetic bug dataset.

## Training Commands

```bash
python -m src.train_optimize_anything --train-size 200 --val-size 50 --test-size 100 --workers 16 --max-metric-calls 300 --run-post-optimization-testset --wandb --repo pygments__pygments

python -m src.train_optimize_anything --train-size 200 --val-size 50 --test-size 100 --workers 16 --max-metric-calls 300 --run-testset --wandb --repo blevesearch__bleve --proposer loop

python -m src.train_optimize_anything --train-size 200 --val-size 50 --test-size 100 --workers 16 --max-metric-calls 300 --run-post-optimization-testset --wandb --repo pallets__jinja
```
