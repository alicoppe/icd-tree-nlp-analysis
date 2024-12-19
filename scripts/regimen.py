import os
import sys

import torch
import torch.optim as optim
from tqdm import tqdm

from scripts.data import create_batch_input_and_labels
import numpy as np

# from data import create_batch_input_and_labels


#### Need to update path saving for the probe ### 

class ProbeRegimen:
    """
    Training regimen for probes using ICD_tree-based distances.

    Attributes:
        optimizer: The optimizer used to train the probe.
        scheduler: The scheduler used to adjust the learning rate during training.
        icd_tree: An instance of the ICD_tree class, used to compute pairwise distances.
    """

    def __init__(self, icd_tree, epochs=10):
        self.icd_tree = icd_tree
        self.max_epochs = epochs

    def set_optimizer(self, probe):
        self.optimizer = optim.Adam(probe.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=0)

    def train(self, probe, loss, train_icd_codes, dev_icd_codes, embedding_dict, batch_size=50, save_path="probe_params.pt"):
        self.set_optimizer(probe)
        min_dev_loss = sys.maxsize
        min_dev_loss_epoch = -1

        for epoch_index in range(self.max_epochs):
            epoch_train_loss = 0
            epoch_dev_loss = 0
            epoch_train_loss_count = 0
            epoch_dev_loss_count = 0

            # Shuffle the training ICD codes for each epoch
            np.random.shuffle(train_icd_codes)

            # Training loop
            with tqdm(total=len(train_icd_codes), desc=f"[epoch {epoch_index}] Training", unit="samples") as pbar_train:
                for batch_start in range(0, len(train_icd_codes), batch_size):
                    batch_end = min(batch_start + batch_size, len(train_icd_codes))
                    train_batch = train_icd_codes[batch_start:batch_end]

                    probe.train()
                    self.optimizer.zero_grad()

                    # Prepare batch inputs
                    batch_embeddings, label_matrix = create_batch_input_and_labels(
                        train_batch,
                        self.icd_tree,
                        embedding_dict
                    )

                    # Forward pass
                    predictions = probe(batch_embeddings)
                    batch_loss = loss(predictions, label_matrix)

                    # Backward pass and optimizer step
                    batch_loss.backward()
                    epoch_train_loss += batch_loss.item()
                    epoch_train_loss_count += 1
                    self.optimizer.step()

                    # Update progress bar
                    pbar_train.update(len(train_batch))

            # Validation loop
            with tqdm(total=len(dev_icd_codes), desc=f"[epoch {epoch_index}] Validation", unit="samples") as pbar_dev:
                for batch_start in range(0, len(dev_icd_codes), batch_size):
                    batch_end = min(batch_start + batch_size, len(dev_icd_codes))
                    dev_batch = dev_icd_codes[batch_start:batch_end]

                    probe.eval()

                    # Prepare batch inputs
                    batch_embeddings, label_matrix = create_batch_input_and_labels(
                        dev_batch,
                        self.icd_tree,
                        embedding_dict
                    )

                    # Forward pass (validation)
                    with torch.no_grad():
                        predictions = probe(batch_embeddings)
                        batch_loss = loss(predictions, label_matrix)

                    epoch_dev_loss += batch_loss.item()
                    epoch_dev_loss_count += 1

                    # Update progress bar
                    pbar_dev.update(len(dev_batch))

            # Update learning rate scheduler
            self.scheduler.step(epoch_dev_loss)

            # Logging and checkpointing
            train_loss_avg = epoch_train_loss / epoch_train_loss_count
            dev_loss_avg = epoch_dev_loss / epoch_dev_loss_count
            tqdm.write(f'[epoch {epoch_index}] Train loss: {train_loss_avg}, Dev loss: {dev_loss_avg}')

            if dev_loss_avg < min_dev_loss - 0.0001:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure directory exists
                torch.save(probe.state_dict(), save_path)  # Save the model
                min_dev_loss = dev_loss_avg
                min_dev_loss_epoch = epoch_index
                tqdm.write(f'Saving probe parameters to {save_path}')
            elif min_dev_loss_epoch < epoch_index - 4:
                tqdm.write('Early stopping')
                break



    def predict(self, probe, icd_code_array, embedding_dict, batch_size=50):
        all_predictions = []

        for batch_start in tqdm(range(0, len(icd_code_array), batch_size), desc='[predicting batches]'):
            batch_end = min(batch_start + batch_size, len(icd_code_array))
            batch = icd_code_array[batch_start:batch_end]

            probe.eval()

            # Prepare batch inputs
            batch_embeddings, _ = create_batch_input_and_labels(
                batch,
                self.icd_tree,
                embedding_dict
            )

            # Generate predictions
            with torch.no_grad():
                predictions = probe(batch_embeddings)
                all_predictions.append(predictions)
            
        return all_predictions