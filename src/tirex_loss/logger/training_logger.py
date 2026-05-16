import json
import polars as pl
from pathlib import Path
from datetime import datetime
from io import StringIO
from typing import Optional


class TrainingLogger:
    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize log files
            self.hparams_file = self.log_dir / 'hyperparameters.json'
            self.epoch_metrics_file = self.log_dir / 'epoch_metrics.jsonl'
            self.batch_metrics_file = self.log_dir / 'batch_metrics.jsonl'
        else:
            self.hparams_file = None
            self.epoch_metrics_file = None
            self.batch_metrics_file = None
        
        # In-memory storage
        self.metrics = {}
        
    def log_hyperparameters(self, hparams: dict):
        """Log hyperparameters to JSON file"""
        if not self.hparams_file:
            raise ValueError("log_dir must be set to log hyperparameters")
        
        with open(self.hparams_file, 'w') as f:
            json.dump(hparams, f, indent=2)
    
    def log_epoch_metrics(self, epoch: int, metrics: dict):
        """Log per-epoch metrics (append-only)"""
        if not self.epoch_metrics_file:
            raise ValueError("log_dir must be set to log epoch metrics")
        
        record = {
            'epoch': epoch, 
            'timestamp': datetime.now().isoformat()
        }
        record.update(metrics)
        
        # Append single line to JSONL
        with open(self.epoch_metrics_file, 'a') as f:
            f.write(json.dumps(record) + '\n')
    
    def log_batch_metrics(self, epoch: int, batch: int, global_step: int, metrics: dict):
        """Log per-batch metrics (append-only)"""
        if not self.batch_metrics_file:
            raise ValueError("log_dir must be set to log batch metrics")
        
        record = {
            'epoch': epoch, 
            'batch': batch,
            'global_step': global_step,
            'timestamp': datetime.now().isoformat()
        }
        record.update(metrics)
        
        # Append single line to JSONL
        with open(self.batch_metrics_file, 'a') as f:
            f.write(json.dumps(record) + '\n')
    
    def flush(self):
        """No-op for JSONL (files are automatically flushed)"""
        pass
    
    def load_metrics_file(self) -> dict:
        """Load all metrics from files or return cached metrics from bytes"""
        # If metrics already loaded from bytes, return them
        if self.metrics:
            return self.metrics
        
        # Otherwise load from files
        if not self.log_dir:
            raise ValueError("Either load_from_bytes() must be called or log_dir must be set")
        
        self.metrics = {}
        # Load hyperparameters
        if self.hparams_file is not None and self.hparams_file.exists():
            with open(self.hparams_file, 'r') as f:
                self.metrics['hyperparameters'] = json.load(f)
        else:
            self.metrics['hyperparameters'] = {}
        
        # Load epoch metrics
        self.metrics['epoch_metrics'] = self.load_epoch_metrics_df()
        
        # Load batch metrics
        self.metrics['batch_metrics'] = self.load_batch_metrics_df()
        
        return self.metrics
    
    def load_epoch_metrics_df(self) -> pl.DataFrame:
        """Load epoch metrics as Polars DataFrame"""
        if self.epoch_metrics_file is None or not self.epoch_metrics_file.exists():
            return pl.DataFrame()
        return pl.read_ndjson(self.epoch_metrics_file)

    def load_batch_metrics_df(self) -> pl.DataFrame:
        """Load batch metrics as Polars DataFrame"""
        if self.batch_metrics_file is None or not self.batch_metrics_file.exists():
            return pl.DataFrame()
        return pl.read_ndjson(self.batch_metrics_file)
    
    def load_metrics_bytes(
        self,
        hparams_bytes: Optional[bytes] = None,
        epoch_metrics_bytes: Optional[bytes] = None,
        batch_metrics_bytes: Optional[bytes] = None
    ):
        """
        Load metrics from byte objects (e.g., from S3)
        
        Args:
            hparams_bytes: Bytes of hyperparameters.json file
            epoch_metrics_bytes: Bytes of epoch_metrics.jsonl file
            batch_metrics_bytes: Bytes of batch_metrics.jsonl file
        """
        self.metrics = {}
        # Load hyperparameters
        if hparams_bytes:
            try:
                self.metrics['hyperparameters'] = json.loads(hparams_bytes.decode('utf-8'))
            except Exception as e:
                print(f"Error loading hyperparameters from bytes: {e}")
                self.metrics['hyperparameters'] = {}
        else:
            self.metrics['hyperparameters'] = {}
        
        # Load epoch metrics
        if epoch_metrics_bytes:
            text = epoch_metrics_bytes.decode('utf-8')
            self.metrics['epoch_metrics'] = pl.read_ndjson(StringIO(text))
        else:
            self.metrics['epoch_metrics'] = pl.DataFrame()
        
        # Load batch metrics
        if batch_metrics_bytes:
            text = batch_metrics_bytes.decode('utf-8')
            self.metrics['batch_metrics'] = pl.read_ndjson(StringIO(text))
        else:
            self.metrics['batch_metrics'] = pl.DataFrame()
        
        return self.metrics
    
    def get_summary(self) -> dict:
        """Get training summary statistics"""
        epoch_df = self.metrics.get('epoch_metrics', None)
        if epoch_df is None:
            return {}
        
        summary = {
            'total_epochs': len(epoch_df),
            'best_train_loss': None,
            'best_test_loss': None,
            'final_train_loss': None,
            'final_test_loss': None
        }
        
        if 'train_loss' in epoch_df.columns:
            summary['best_train_loss'] = float(epoch_df['train_loss'].min())
            summary['final_train_loss'] = float(epoch_df['train_loss'][-1])
        
        if 'test_loss' in epoch_df.columns:
            summary['best_test_loss'] = float(epoch_df['test_loss'].min())
            summary['final_test_loss'] = float(epoch_df['test_loss'][-1])
        
        return summary