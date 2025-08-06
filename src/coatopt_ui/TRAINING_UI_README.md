# PC-HPPO-OML Training UI

A graphical user interface for real-time monitoring of PC-HPPO-OML coating optimization training.

## Features

- **Configuration Loading**: Easy loading of INI configuration files
- **Real-time Training Monitoring**: Live updates during training process
- **Reward Visualization**: Interactive plots showing training rewards over time with moving averages
- **Pareto Front Evolution**: Visualization of how the Pareto front evolves during training
- **Progress Tracking**: Visual progress bars and status updates

## Usage

### Quick Start

1. **Launch the UI**:
   ```bash
   coatopt-ui
   ```

2. **Load Configuration**:
   - Click "Browse" to select your configuration INI file
   - Click "Load Configuration" to initialize the training components
   - Wait for confirmation that materials and environment are loaded

3. **Start Training**:
   - Click "Start Training" to begin the optimization process
   - Monitor progress in real-time through the plot tabs
   - Use "Stop Training" if you need to halt the process

### UI Components

#### Control Panel
- **Configuration File**: Browse and select your training configuration
- **Load Configuration**: Initialize training components from config file
- **Start/Stop Training**: Control the training process
- **Status Display**: Current training status and progress information
- **Progress Bar**: Visual indication of training activity
- **Epoch scan**: Scan though the epochs to see how the pareto front is changing

#### Plot Tabs

##### Training Rewards Tab
- **Real-time Updates**: Shows reward evolution during training
- **Moving Average**: Smoothed trend line for better visualization
- **Episode Tracking**: X-axis shows training episodes, Y-axis shows total reward

###### Training Values Tab

 - View the values of reflectivity, absorption, thickness and thermal noise through training 

##### Pareto Front Evolution Tab
- **Multi-objective Visualization**: Shows Pareto front in reflectivity vs absorption space
- **Evolution Over Time**: Different colors show front evolution through training
- **Target Lines**: Red dashed lines indicate optimization targets
- **Log Scale**: Both axes use logarithmic scaling for better visualization
- **Interactive**: Latest front highlighted with larger markers

## Configuration Requirements

The UI works with standard PC-HPPO-OML configuration files. Ensure your INI file includes:

### Required Sections
```ini
[General]
root_dir = /path/to/output/directory
materials_file = /path/to/materials.json

[Data]
n_layers = 20
min_thickness = 0.01
max_thickness = 0.5
optimise_parameters = ["reflectivity", "absorption"]
optimise_targets = {"reflectivity": 0.99999, "absorption": 10}

[Training]
n_iterations = 2000
n_training_epochs = 10
beta_start = 1.0
beta_end = 0.001
```

## Technical Details

### Threading
- Training runs in a separate thread to keep UI responsive
- Queue-based communication between training and UI threads
- Non-blocking updates ensure smooth visualization

### Data Processing
- Real-time metric collection from training loop
- Pareto front updates every 50 episodes (configurable)
- Automatic data buffering for plot history

### Plot Updates
- **Rewards Plot**: Updates every episode with latest reward data
- **Pareto Plot**: Updates every 50 episodes with front evolution
- **Performance**: Optimized for real-time updates without blocking

## Troubleshooting

### Common Issues

1. **Import Errors**: 
   - Ensure all dependencies are installed: `pip install matplotlib pandas numpy pymoo`
   - Check that coatopt package is in Python path

2. **Configuration Loading Fails**:
   - Verify INI file format is correct
   - Check that materials file path exists and is accessible
   - Ensure all required configuration sections are present

3. **Training Won't Start**:
   - Load configuration first before starting training
   - Check that output directory is writable
   - Verify configuration has valid parameters

4. **Plots Not Updating**:
   - Training must be actively running for live updates
   - Check console for error messages
   - Restart UI if plots freeze

### Performance Tips

- Use reasonable iteration counts for responsive UI (start with 1000-2000)
- Close other applications for better performance during training
- Monitor memory usage for long training runs
- Save configuration presets for repeated use

## Integration with Existing Workflows

The UI integrates seamlessly with existing PC-HPPO-OML workflows:

- Uses same configuration files as command-line training
- Produces identical output files and checkpoints
- Compatible with existing analysis and evaluation scripts
- Supports continue training from previous checkpoints

## Example Workflow

1. **Prepare Configuration**: Create or modify existing INI file
2. **Launch UI**: Run `python launch_ui.py`
3. **Load Config**: Browse to your INI file and load
4. **Monitor Training**: Start training and observe real-time progress
5. **Analyze Results**: Use generated outputs with existing analysis tools

The UI provides an intuitive way to monitor training progress while maintaining full compatibility with the underlying PC-HPPO-OML algorithm.