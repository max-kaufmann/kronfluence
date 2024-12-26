import argparse
import logging
import os
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from accelerate.utils import set_seed
from torch import nn
from torch.optim import lr_scheduler
from torch.utils import data
from examples.mnist.pipeline import get_mnist_dataset, construct_mnist_classifier
import time
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a small model on the MNIST dataset.")

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
        help="A folder to download or load CIFAR-10 dataset.",
    )

    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=512,
        help="Batch size for the training dataloader.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1024,
        help="Batch size for the evaluation dataloader.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.4,
        help="Initial learning rate to train the model.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.001,
        help="Weight decay to train the model.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=25,
        help="Total number of epochs to train the model.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1004,
        help="A seed for reproducible training pipeline.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="A path to store the final checkpoint.",
    )
    parser.add_argument(
        "--dataset_in_memory",
        action=argparse.BooleanOptionalAction,
        help="Whether to have the dataset in memory, instead of on disk.",
    )
    args = parser.parse_args()

    if args.checkpoint_dir is not None:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    return args


def train(
    dataset: data.Dataset,
    batch_size: int,
    num_train_epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> nn.Module:
    print("Training starting", time.time())
    train_dataloader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    print("Loading model and optimizer!")
    model = construct_mnist_classifier().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    print("Loaded!")

    iters_per_epoch = len(train_dataloader)
    lr_peak_epoch = num_train_epochs // 5
    lr_schedule = np.interp(
        np.arange((num_train_epochs + 1) * iters_per_epoch),
        [0, lr_peak_epoch * iters_per_epoch, num_train_epochs * iters_per_epoch],
        [0, 1, 0],
    )
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)

    start_time = time.time()
    model.train()
    for epoch in tqdm(range(num_train_epochs)):

        # run your training step here
        total_loss = 0.0
        (
            data_loading_time_avg,
            moving_to_gpu_time_avg,
            forward_pass_time_avg,
            backward_pass_time_avg,
            optimization_time_avg,
            batch_processing_time_avg,
        ) = 0, 0, 0, 0, 0,0
        batch_start_time = time.time()

        for batch in tqdm(train_dataloader):
            # Data loading time (includes moving to device)
            data_loading_end_time = time.time()
            inputs, labels = batch

            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            moving_to_gpu_time = time.time()

            # Forward pass
            forward_pass_start_time = time.time()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            forward_pass_end_time = time.time()

            # Backward pass
            backward_pass_start_time = time.time()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            backward_pass_end_time = time.time()

            # Optimization
            optimization_start_time = time.time()
            optimizer.step()
            scheduler.step()
            optimization_end_time = time.time()

            # Calculate times
            data_loading_time_avg += data_loading_end_time - batch_start_time
            moving_to_gpu_time_avg += moving_to_gpu_time - data_loading_end_time
            forward_pass_time_avg += forward_pass_end_time - forward_pass_start_time
            backward_pass_time_avg += backward_pass_end_time - backward_pass_start_time
            optimization_time_avg += optimization_end_time - optimization_start_time
            batch_processing_time_avg += time.time() - batch_start_time

            total_loss += loss.detach().float()
            batch_start_time = time.time()
        
        # print(prof.key_averages(group_by_stack_n=5).table(sort_by="cuda_time_total",row_limit=10))

        print(
            f"Data loading time: {data_loading_time_avg / len(train_dataloader)}, Moving to GPU time: {moving_to_gpu_time_avg / len(train_dataloader)}, Forward pass time: {forward_pass_time_avg / len(train_dataloader)}, Backward pass time: {backward_pass_time_avg / len(train_dataloader)}, Optimization time: {optimization_time_avg / len(train_dataloader)}, Batch processing time: {batch_processing_time_avg / len(train_dataloader)}"
        )
        # print(prof.key_averages().table(sort_by="cuda_time_total"))
        logging.info(f"Epoch {epoch + 1} - Averaged Loss: {total_loss / len(dataset)}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Completed training in {elapsed_time:.2f} seconds.")
    return model


def evaluate(model: nn.Module, dataset: data.Dataset, batch_size: int) -> Tuple[float, float]:
    dataloader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    model.eval()
    total_loss, total_correct = 0.0, 0
    for batch in dataloader:
        with torch.no_grad():
            inputs, labels = batch
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels, reduction="sum")
            total_loss += loss.detach().float()
            total_correct += outputs.detach().argmax(1).eq(labels).sum()

    return total_loss.item() / len(dataloader.dataset), total_correct.item() / len(dataloader.dataset)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    if args.seed is not None:
        set_seed(args.seed)

    train_dataset = get_mnist_dataset(split="train", in_memory=args.dataset_in_memory)
    model = train(
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    eval_train_dataset = get_mnist_dataset(split="train", dataset_dir=args.dataset_dir, in_memory=args.dataset_in_memory)
    train_loss, train_acc = evaluate(model=model, dataset=eval_train_dataset, batch_size=args.eval_batch_size)
    logger.info(f"Train loss: {train_loss}, Train Accuracy: {train_acc}")

    eval_dataset = get_mnist_dataset(split="test", dataset_dir=args.dataset_dir, in_memory=args.dataset_in_memory)
    eval_loss, eval_acc = evaluate(model=model, dataset=eval_dataset, batch_size=args.eval_batch_size)
    logger.info(f"Evaluation loss: {eval_loss}, Evaluation Accuracy: {eval_acc}")

    if args.checkpoint_dir is not None:
        model_name = "model"
        if args.corrupt_percentage is not None:
            model_name += "_corrupt_" + str(args.corrupt_percentage)
        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"{model_name}.pth"))


if __name__ == "__main__":
    main()
