"""
Unified Training Checkpoint Manager for HPPO Trainer

This module provides a consistent, efficient way to save and load all training data
using HDF5 format with optional compression and structured organization.

Benefits:
- Single file format for all numerical data
- Efficient compression and fast I/O
- Hierarchical organization
- Version control and metadata support
- Cross-platform compatibility
- Easy integration with analysis tools
"""

import os
import h5py
import json
import pickle
import numpy as np
import pandas as pd
import pandas as pd
import torch
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime


class TrainingCheckpointManager:
    """
    Manages saving and loading of all training-related data in a unified format.
    
    Structure:
    training_checkpoint.h5                     # Main checkpoint file (fast, compressed)
    ├── metadata/
    │   ├── training_config (JSON string)
    │   ├── environment_config (JSON string) 
    │   ├── creation_time
    │   └── last_updated
    ├── training_data/
    │   ├── metrics (structured array from pandas DataFrame)
    │   ├── episode_rewards
    │   ├── episode_times
    │   └── objective_weights_history
    ├── pareto_data/
    │   ├── current_front
    │   ├── all_points
    │   ├── all_values  
    │   ├── reference_point
    │   └── fronts_history/
    │       ├── episode_XXXX
    │       └── ...
    ├── environment_state/
    │   ├── pareto_front
    │   ├── saved_points
    │   ├── saved_data
    │   └── optimization_parameters
    └── best_states/
        └── states_data (pickle-serialized for complex structures)
    
    model_weights/                             # Separate directory for .pt files
    ├── discrete_policy.pt
    ├── continuous_policy.pt
    └── value.pt
    
    Note: Network weights saved as separate .pt files to keep HDF5 file small and fast.
    Plots and visualizations are optional (save_plots=False by default).
    ├── environment_state/
    │   ├── pareto_front
    │   ├── saved_points
    │   ├── saved_data
    │   └── optimization_parameters
    ├── model_weights/
    │   └── [saved as separate .pt files - keep existing PyTorch format]
    └── best_states/
        └── states_data (pickle-serialized for complex structures)
    
    PNG files (plots, visualizations) remain as separate files for easy viewing.
    """
    
    def __init__(self, root_dir: str, checkpoint_name: str = "training_checkpoint.h5"):
        """
        Initialize checkpoint manager.
        
        Args:
            root_dir: Root directory for training outputs
            checkpoint_name: Name of the HDF5 checkpoint file
        """
        self.root_dir = root_dir
        self.checkpoint_path = os.path.join(root_dir, checkpoint_name)
        self.backup_path = os.path.join(root_dir, f"backup_{checkpoint_name}")
        
        # Ensure root directory exists
        os.makedirs(root_dir, exist_ok=True)
    
    def save_complete_checkpoint(self, trainer_data: Dict[str, Any]) -> None:
        """
        Save complete training checkpoint with atomic write operation.
        
        Args:
            trainer_data: Dictionary containing all trainer data to save
        """
        # Backup existing file first
        if os.path.exists(self.checkpoint_path):
            if os.path.exists(self.backup_path):
                os.remove(self.backup_path)
            os.rename(self.checkpoint_path, self.backup_path)
        
        try:
            with h5py.File(self.checkpoint_path, 'w') as f:
                # Save metadata
                self._save_metadata(f, trainer_data.get('metadata', {}))
                
                # Save training data
                self._save_training_data(f, trainer_data.get('training_data', {}))
                
                # Save Pareto data
                self._save_pareto_data(f, trainer_data.get('pareto_data', {}))
                
                # Save environment state
                self._save_environment_state(f, trainer_data.get('environment_state', {}))
                
                # Save best states (complex Python objects)
                #self._save_best_states(f, trainer_data.get('best_states', []))
                
        except Exception as e:
            # Restore backup if save failed
            if os.path.exists(self.backup_path):
                if os.path.exists(self.checkpoint_path):
                    os.remove(self.checkpoint_path)
                os.rename(self.backup_path, self.checkpoint_path)
            raise e
        
        # Remove backup after successful save
        if os.path.exists(self.backup_path):
            os.remove(self.backup_path)
    
    def load_complete_checkpoint(self) -> Dict[str, Any]:
        """
        Load complete training checkpoint.
        
        Returns:
            Dictionary containing all loaded trainer data
        """
        if not os.path.exists(self.checkpoint_path):
            return {}
        
        data = {}
        
        try:
            with h5py.File(self.checkpoint_path, 'r') as f:
                data['metadata'] = self._load_metadata(f)
                data['training_data'] = self._load_training_data(f)
                data['pareto_data'] = self._load_pareto_data(f)
                data['environment_state'] = self._load_environment_state(f)
                #data['best_states'] = self._load_best_states(f)
        
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return {}
        
        return data
    
    def _save_metadata(self, h5file: h5py.File, metadata: Dict[str, Any]) -> None:
        """Save metadata group."""
        meta_group = h5file.create_group('metadata')
        
        # Save timestamps
        meta_group.create_dataset('creation_time', data=datetime.now().isoformat())
        meta_group.create_dataset('last_updated', data=datetime.now().isoformat())
        
        # Save configurations as JSON strings
        if 'training_config' in metadata:
            meta_group.create_dataset('training_config', 
                                    data=json.dumps(metadata['training_config']))
        
        if 'environment_config' in metadata:
            meta_group.create_dataset('environment_config', 
                                    data=json.dumps(metadata['environment_config']))
        
        # Save other metadata
        for key, value in metadata.items():
            if key not in ['training_config', 'environment_config']:
                if isinstance(value, (str, int, float)):
                    meta_group.create_dataset(key, data=value)
    
    def _save_training_data(self, h5file: h5py.File, training_data: Dict[str, Any]) -> None:
        """Save training data group with compression."""
        train_group = h5file.create_group('training_data')
        
        # Save metrics DataFrame as structured array
        if 'metrics_df' in training_data:
            df = training_data['metrics_df']
            
            # Clean DataFrame to ensure HDF5 compatibility
            df_clean = df.copy()
            
            # Convert object columns to appropriate types
            for col in df_clean.columns:
                if df_clean[col].dtype == 'object':
                    # Replace empty strings and NaN with appropriate defaults
                    if df_clean[col].isnull().all():
                        # All null - convert to float and fill with 0
                        df_clean[col] = 0.0
                    else:
                        # Try to convert to numeric, fall back to string
                        try:
                            # Replace empty strings with NaN first
                            df_clean[col] = df_clean[col].replace('', np.nan)
                            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0.0)
                        except:
                            # Convert to string and handle empty values
                            df_clean[col] = df_clean[col].astype(str).replace('nan', '').replace('', 'none')
            
            # Convert DataFrame to structured array for HDF5 storage
            structured_data = df_clean.to_records(index=False)
            train_group.create_dataset('metrics', 
                                     data=structured_data, 
                                     compression='gzip', 
                                     compression_opts=9)
        
        # Save other arrays with compression
        for key, value in training_data.items():
            if key != 'metrics_df' and isinstance(value, np.ndarray):
                train_group.create_dataset(key, 
                                         data=value, 
                                         compression='gzip', 
                                         compression_opts=6)
    
    def _save_pareto_data(self, h5file: h5py.File, pareto_data: Dict[str, Any]) -> None:
        """Save Pareto front data with history."""
        pareto_group = h5file.create_group('pareto_data')
        
        # Save current Pareto data
        for key, value in pareto_data.items():
            if key != 'fronts_history' and isinstance(value, np.ndarray):
                pareto_group.create_dataset(key, 
                                          data=value, 
                                          compression='gzip', 
                                          compression_opts=6)
            elif not isinstance(value, np.ndarray):
                print("WARNING: Skipping non-array Pareto data key:", key)
        
        # Save historical Pareto fronts
        if 'fronts_history' in pareto_data:
            history_group = pareto_group.create_group('fronts_history')
            for episode, front_data in pareto_data['fronts_history'].items():
                if isinstance(front_data, np.ndarray):
                    history_group.create_dataset(f'episode_{episode}', 
                                               data=front_data,
                                               compression='gzip',
                                               compression_opts=6)
    
    def _save_environment_state(self, h5file: h5py.File, env_state: Dict[str, Any]) -> None:
        """Save environment state."""
        env_group = h5file.create_group('environment_state')
        
        for key, value in env_state.items():
            if isinstance(value, np.ndarray):
                env_group.create_dataset(key, 
                                       data=value, 
                                       compression='gzip', 
                                       compression_opts=6)
            elif isinstance(value, list):
                # Convert lists to JSON strings for storage
                env_group.create_dataset(key, data=json.dumps(value))
            elif isinstance(value, (str, int, float)):
                env_group.create_dataset(key, data=value)
    
    def _save_best_states(self, h5file: h5py.File, best_states: List) -> None:
        """Save best states using pickle for complex structures."""
        states_group = h5file.create_group('best_states')
        
        # Serialize complex Python objects with pickle
        if best_states:
            pickled_data = pickle.dumps(best_states)
            states_group.create_dataset('states_data', 
                                      data=np.frombuffer(pickled_data, dtype=np.uint8))
    
    def _load_metadata(self, h5file: h5py.File) -> Dict[str, Any]:
        """Load metadata group."""
        if 'metadata' not in h5file:
            return {}
        
        meta_group = h5file['metadata']
        metadata = {}
        
        for key in meta_group.keys():
            value = meta_group[key][()]
            if isinstance(value, bytes):
                value = value.decode('utf-8')
            
            # Parse JSON configurations
            if key in ['training_config', 'environment_config']:
                try:
                    metadata[key] = json.loads(value)
                except:
                    metadata[key] = value
            else:
                metadata[key] = value
        
        return metadata
    
    def _load_training_data(self, h5file: h5py.File) -> Dict[str, Any]:
        """Load training data group."""
        if 'training_data' not in h5file:
            return {}
        
        train_group = h5file['training_data']
        training_data = {}
        
        for key in train_group.keys():
            dataset = train_group[key]
            
            # Handle scalar datasets
            if dataset.shape == ():
                data = dataset[()]  # Use [()] for scalar datasets
            else:
                data = dataset[:]   # Use [:] for array datasets
            
            # Convert structured array back to DataFrame
            if key == 'metrics':
                training_data['metrics_df'] = pd.DataFrame(data)
            else:
                training_data[key] = data
        
        return training_data
    
    def _load_pareto_data(self, h5file: h5py.File) -> Dict[str, Any]:
        """Load Pareto data group."""
        if 'pareto_data' not in h5file:
            return {}
        
        pareto_group = h5file['pareto_data']
        pareto_data = {}
        
        for key in pareto_group.keys():
            if key == 'fronts_history':
                # Load historical fronts
                history_group = pareto_group[key]
                pareto_data['fronts_history'] = {}
                for episode_key in history_group.keys():
                    episode = int(episode_key.split('_')[1])
                    dataset = history_group[episode_key]
                    if dataset.shape == ():
                        pareto_data['fronts_history'][episode] = dataset[()]
                    else:
                        pareto_data['fronts_history'][episode] = dataset[:]
            else:
                dataset = pareto_group[key]
                if dataset.shape == ():
                    pareto_data[key] = dataset[()]
                else:
                    pareto_data[key] = dataset[:]
        
        return pareto_data
    
    def _load_environment_state(self, h5file: h5py.File) -> Dict[str, Any]:
        """Load environment state group."""
        if 'environment_state' not in h5file:
            return {}
        
        env_group = h5file['environment_state']
        env_state = {}
        
        for key in env_group.keys():
            dataset = env_group[key]
            
            if dataset.dtype.kind == 'S':  # String data
                if dataset.shape == ():
                    value = dataset[()].decode('utf-8')
                else:
                    value = dataset[:].decode('utf-8')
                # Try to parse as JSON
                try:
                    env_state[key] = json.loads(value)
                except:
                    env_state[key] = value
            else:
                if dataset.shape == ():
                    env_state[key] = dataset[()]
                else:
                    env_state[key] = dataset[:]
        
        return env_state
    
    def _load_best_states(self, h5file: h5py.File) -> List:
        """Load best states from pickle data."""
        if 'best_states' not in h5file or 'states_data' not in h5file['best_states']:
            return []
        
        states_group = h5file['best_states']
        pickled_bytes = states_group['states_data'][:].tobytes()
        
        try:
            return pickle.loads(pickled_bytes)
        except:
            return []
    
    def get_checkpoint_info(self) -> Dict[str, Any]:
        """Get basic information about the checkpoint file."""
        if not os.path.exists(self.checkpoint_path):
            return {"exists": False}
        
        stat_info = os.stat(self.checkpoint_path)
        
        info = {
            "exists": True,
            "size_bytes": stat_info.st_size,
            "size_mb": round(stat_info.st_size / (1024 * 1024), 2),
            "modified_time": datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
        }
        
        # Try to get internal structure info
        try:
            with h5py.File(self.checkpoint_path, 'r') as f:
                info["groups"] = list(f.keys())
                if 'metadata' in f and 'last_updated' in f['metadata']:
                    info["last_updated"] = f['metadata']['last_updated'][()].decode('utf-8')
        except:
            pass
        
        return info
