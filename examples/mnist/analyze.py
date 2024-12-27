import argparse
import logging
import os
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from examples.mnist.pipeline import get_mnist_dataset, construct_mnist_classifier
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments, ScoreArguments
from kronfluence.task import Task
from kronfluence.utils.common.factor_arguments import all_low_precision_factor_arguments
from kronfluence.utils.common.score_arguments import all_low_precision_score_arguments
from kronfluence.utils.dataset import DataLoaderKwargs
from examples.mnist.pipeline import add_box_to_mnist_dataset

BATCH_TYPE = Tuple[torch.Tensor, torch.Tensor]


def parse_args():
    parser = argparse.ArgumentParser(description="Influence analysis on CIFAR-10 dataset.")

    parser.add_argument(
        "--corrupt_percentage",
        type=float,
        default=None,
        help="Percentage of the training dataset to corrupt.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./data",
        help="A folder to download or load the MNIST dataset.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="A path to the checkpoint which is being evaluated",
    )
    parser.add_argument(
        "--factor_strategy",
        type=str,
        default="ekfac",
        help="Strategy to compute influence factors.",
    )
    parser.add_argument(
        "--query_batch_size",
        type=int,
        default=1000,
        help="Batch size for computing query gradients.",
    )
    parser.add_argument(
        "--use_half_precision",
        action="store_true",
        default=False,
        help="Whether to use half precision for computing factors and scores.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="Boolean flag to profile computations.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.model_path):
        raise ValueError(f"No checkpoint found at {args.model_path}.")

    return args


class ClassificationTask(Task):
    def compute_train_loss(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:
        inputs, labels = batch
        logits = model(inputs)
        if not sample:
            return F.cross_entropy(logits, labels, reduction="sum")
        with torch.no_grad():
            probs = torch.nn.functional.softmax(logits.detach(), dim=-1)
            sampled_labels = torch.multinomial(
                probs,
                num_samples=1,
            ).flatten()
        return F.cross_entropy(logits, sampled_labels, reduction="sum")

    def compute_measurement(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
    ) -> torch.Tensor:
        # Copied from: https://github.com/MadryLab/trak/blob/main/trak/modelout_functions.py. Returns the margin between the correct logit and the second most likely prediction
        inputs, labels = batch
        logits = model(inputs)

        # Get correct logit values
        bindex = torch.arange(logits.shape[0]).to(device=logits.device, non_blocking=False)
        logits_correct = logits[bindex, labels]

        # Get the other logits, and take the softmax of them
        cloned_logits = logits.clone()
        cloned_logits[bindex, labels] = torch.tensor(-torch.inf, device=logits.device, dtype=logits.dtype)
        maximum_non_correct_logits = cloned_logits.logsumexp(dim=-1)

        # Look at the  margin, the difference between the correct logits and the (soft) maximum non-correctl logits
        margins = logits_correct - maximum_non_correct_logits
        return -margins.sum()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    # Prepare the dataset.
    train_dataset = get_mnist_dataset(split="eval_train", dataset_dir=args.dataset_dir)
    eval_dataset = get_mnist_dataset(split="test", dataset_dir=args.dataset_dir)

    # Prepare the trained model.
    model = construct_mnist_classifier()
    model.load_state_dict(torch.load(args.model_path))

    # Define task and prepare model.
    task = ClassificationTask()
    model = prepare_model(model, task)

    analyzer = Analyzer(
        analysis_name="mnist",
        model=model,
        task=task,
        profile=args.profile,
    )
    # Configure parameters for DataLoader.
    dataloader_kwargs = DataLoaderKwargs(num_workers=4)
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    # Compute influence factors.
    factors_name = args.factor_strategy
    factor_args = FactorArguments(strategy=args.factor_strategy)
    if args.use_half_precision:
        factor_args = all_low_precision_factor_arguments(strategy=args.factor_strategy, dtype=torch.bfloat16)
        factors_name += "_half"
    analyzer.fit_all_factors(
        factors_name=factors_name,
        factor_args=factor_args,
        dataset=train_dataset,
        per_device_batch_size=None,
        overwrite_output_dir=False,
    )

    # Compute pairwise scores.
    score_args = ScoreArguments()
    scores_name = factor_args.strategy
    if args.use_half_precision:
        score_args = all_low_precision_score_arguments(dtype=torch.bfloat16)
        scores_name += "_half"
    analyzer.compute_pairwise_scores(
        scores_name=scores_name,
        score_args=score_args,
        factors_name=factors_name,
        query_dataset=eval_dataset,
        query_indices=list(range(2000)),
        train_dataset=train_dataset,
        per_device_query_batch_size=args.query_batch_size,
        overwrite_output_dir=False,
    )
    scores = analyzer.load_pairwise_scores(scores_name)["all_modules"]
    logging.info(f"Scores shape: {scores.shape}")


if __name__ == "__main__":
    main()
