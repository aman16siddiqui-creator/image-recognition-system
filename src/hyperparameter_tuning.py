import itertools
import random
import torch
import torch.nn as nn
from src.architectures import DeepAlexNetStyleCNN
from src.trainer import ModelTrainer

class HyperparameterTuner:
    """
    Hyperparameter tuning framework executing Grid Search or Random Search.
    """
    def __init__(self, train_loader_builder, val_loader_builder, num_classes=10):
        self.train_loader_builder = train_loader_builder
        self.val_loader_builder = val_loader_builder
        self.num_classes = num_classes

    def grid_search(self, param_grid, epochs_per_trial=3):
        """
        Executes exhaustive Grid Search across param_grid options.
        """
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        results = []
        best_acc = -1.0
        best_params = None

        print(f"Starting Grid Search with {len(combinations)} parameter configurations...")

        for idx, config in enumerate(combinations, 1):
            print(f"\n--- Trial {idx}/{len(combinations)}: {config} ---")
            train_loader = self.train_loader_builder(config.get('batch_size', 64))
            val_loader = self.val_loader_builder(config.get('batch_size', 64))

            model = DeepAlexNetStyleCNN(num_classes=self.num_classes, dropout_rate=config.get('dropout', 0.4))
            
            lr = config.get('lr', 1e-3)
            weight_decay = config.get('weight_decay', 1e-4)
            opt_name = config.get('optimizer', 'adam').lower()

            if opt_name == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            else:
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

            criterion = nn.CrossEntropyLoss()
            trainer = ModelTrainer(model, optimizer, criterion)
            history = trainer.fit(train_loader, val_loader, epochs=epochs_per_trial, patience=2, save_path=f'models/trial_{idx}.pt')

            final_val_acc = max(history['val_acc'])
            results.append({'config': config, 'val_acc': final_val_acc, 'val_loss': min(history['val_loss'])})

            if final_val_acc > best_acc:
                best_acc = final_val_acc
                best_params = config

        print(f"\nGrid Search Finished! Best Val Acc: {best_acc:.2f}% | Best Config: {best_params}")
        return best_params, best_acc, results

    def random_search(self, param_distributions, n_iter=5, epochs_per_trial=3):
        """
        Executes Random Search over parameter space.
        """
        results = []
        best_acc = -1.0
        best_params = None

        for idx in range(1, n_iter + 1):
            config = {k: random.choice(v) for k, v in param_distributions.items()}
            print(f"\n--- Random Trial {idx}/{n_iter}: {config} ---")

            train_loader = self.train_loader_builder(config.get('batch_size', 64))
            val_loader = self.val_loader_builder(config.get('batch_size', 64))

            model = DeepAlexNetStyleCNN(num_classes=self.num_classes, dropout_rate=config.get('dropout', 0.4))
            optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr', 1e-3), weight_decay=config.get('weight_decay', 1e-4))
            criterion = nn.CrossEntropyLoss()

            trainer = ModelTrainer(model, optimizer, criterion)
            history = trainer.fit(train_loader, val_loader, epochs=epochs_per_trial, patience=2, save_path=f'models/rand_trial_{idx}.pt')

            final_val_acc = max(history['val_acc'])
            results.append({'config': config, 'val_acc': final_val_acc})

            if final_val_acc > best_acc:
                best_acc = final_val_acc
                best_params = config

        return best_params, best_acc, results
