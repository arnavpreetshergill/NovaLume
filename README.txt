1. Final Model Performance

Final test results on the training set:

Dishwasher (Complex, Long Cycle)
F1 Score: 82.05%
Disaggregation Accuracy (DA): 71.28%
NRMSE: 0.0469

Microwave (Sparse, Short Cycle)
F1 Score: 81.84%
Disaggregation Accuracy (DA): 64.79%
NRMSE: 0.0212

Fridge (Cyclic, Medium Cycle)
F1 Score: 75.79%
Disaggregation Accuracy (DA): 68.89%
NRMSE: 0.0149


2. Preprocessing (preprocess_ukdale.py)

The preprocessing script prepares the raw HDF5 data for the neural network.

Downsampling
The native 6 second data is resampled to 10 second intervals. This is a standard resolution that reduces the overall dataset size while maintaining enough granularity to detect sharp appliance power signatures.

Multi Resolution Sequence Windows
Instead of forcing all appliances to share the same sequence length, the script creates tailored window sizes based on the physical runtime of each appliance:

Fridge and Dishwasher: Window size of 599 with a step of 6. This gives the model roughly 1.6 hours of context to capture full multi phase dishwasher cycles and long compressor states. The step of 6 (sliding the window 1 minute at a time) minimizes redundant data points.
Microwave: Window size of 99 with a step of 2. Microwaves run in short bursts. A tight window (roughly 16 minutes) forces the network to focus on the immediate event and prevents the brief microwave signal from being mathematically diluted by surrounding noise.

Sequence to Point Target Alignment
The target value (Y) is aligned to the exact midpoint of the aggregate sequence window (X). This allows the bidirectional model to evaluate past and future context simultaneously.

Normalization
Every appliance and the aggregate signal are normalized.


3. Model Training (train_nnan.py)

The training script utilizes an Inception LSTM hybrid to decode non linear power signatures, supported by a specialized loss function to handle the extreme sparsity of appliance usage.

Model Architecture
Convolutions: 1D convolutions with kernel sizes of 1 and 5, alongside a dilated convolution (kernel 5, dilation 2). This multi scale feature extraction picks up sudden spikes (microwaves) and flat power draws (dishwasher heating elements).
Bi LSTM: A bidirectional LSTM processes the convolutional features to map temporal dependencies.
Softplus Output: The dense layer utilizes a Softplus activation function. This strictly enforces positive wattage outputs (since power cannot be negative) but maintains a continuous gradient to prevent the Dying ReLU problem.

Hybrid Asymmetric Loss Function
Because appliances like dishwashers and microwaves are off most of the day, standard loss functions encourage the model to guess 0W permanently (Zero Collapse). This custom loss combines two metrics:

1. Asymmetric L1 (MAE): Applies a massive multiplier to the error only when the appliance is actually running. This forces the model to prioritize catching the ON state.
2. Standard MSE: Squares the error to force the model to accurately hit the peak physical wattage heights.

Appliance Specific Hyperparameters
The loss function parameters are dynamically assigned based on the appliance's specific behavioral profile:

Fridge and Dishwasher
ON Penalty is 20.0x and MSE Weight is 2.0
The massive 20x penalty terrifies the model into predicting the ON state (breaking Zero Collapse). The moderate MSE weight shapes the peaks without causing stage fright on massive 2000W spikes.

Microwave
ON Penalty is 8.0x and MSE Weight is 15.0
Because the 99 step window makes timing easy, the ON penalty is lowered to prevent paranoid Phantom Power false positives. The MSE weight is increased to 15.0, forcing the network to aggressively recreate the extreme 1600W amplitude natively.

Optimizer and Callbacks
Optimizer: AdamW handles noisy time series data well by decoupling weight decay.
Scheduler: ReduceLROnPlateau cuts the learning rate by half with a patience of 3 epochs if the validation loss stagnates, allowing the model to make micro adjustments late in the training cycle.
Standby Clamping: During final evaluation, predictions below standby noise thresholds (15W for fridge, 10W for dishwasher, 50W for microwave) are clamped to 0W to calculate clean final metrics.
