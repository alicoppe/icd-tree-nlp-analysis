import os
import sys

import torch
import torch.optim as optim
from tqdm import tqdm
import time

from data import create_batch_input_and_labels


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

    def train(self, probe, loss, train_icd_codes, dev_icd_codes, embedding_dict, batch_size=10, name="probe"):
        self.set_optimizer(probe)
        min_dev_loss = sys.maxsize
        min_dev_loss_epoch = -1

        for epoch_index in range(self.max_epochs):
            epoch_train_loss = 0
            epoch_dev_loss = 0
            epoch_train_loss_count = 0
            epoch_dev_loss_count = 0

            # Create and shuffle training batches
            train_batches = [train_icd_codes[i:i + batch_size] for i in range(0, len(train_icd_codes), batch_size)]
            train_batches = [tb for tb in train_batches if len(tb) > 1]
            np.random.shuffle(train_batches)

            # Training loop
            with tqdm(total=len(train_icd_codes), desc=f"[epoch {epoch_index}] Training", unit="samples") as pbar_train:
                for train_batch in train_batches:
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
            dev_batches = [dev_icd_codes[i:i + batch_size] for i in range(0, len(dev_icd_codes), batch_size)]
            dev_batches = [db for db in dev_batches if len(db) > 1]

            with tqdm(total=len(dev_icd_codes), desc=f"[epoch {epoch_index}] Validation", unit="samples") as pbar_dev:
                for dev_batch in dev_batches:
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
                min_dev_loss = dev_loss_avg
                min_dev_loss_epoch = epoch_index

                # Save the model parameters
                save_path = f"saved_models/{name}_batch_size_{batch_size}_epoch_{epoch_index}.pt"
                torch.save(probe.state_dict(), save_path)
                tqdm.write(f'[epoch {epoch_index}] New best model saved at: {save_path}')

            elif min_dev_loss_epoch < epoch_index - 4:
                tqdm.write('Early stopping')
                break


    def predict(self, probe, icd_code_array, embedding_dict, batch_size=50):
        all_predictions = []
        label_distances = []

        # Create and shuffle prediction batches
        prediction_batches = [icd_code_array[i:i + batch_size] for i in range(0, len(icd_code_array), batch_size)]
        prediction_batches = [pb for pb in prediction_batches if len(pb) > 1]

        for batch in tqdm(prediction_batches, desc='[predicting batches]'):
            probe.eval()

            # Prepare batch inputs
            batch_embeddings, label_distance = create_batch_input_and_labels(
                batch,
                self.icd_tree,
                embedding_dict
            )

            label_distances.append(label_distance)

            # Generate predictions
            with torch.no_grad():
                predictions = probe(batch_embeddings)
                all_predictions.append(predictions)

        return all_predictions, label_distances
