import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from layer import Layer_Dense, Activation_ReLU, Activation_Softmax_Loss_CategoricalCrossentropy           

# Training with multiple improvements
nnfs.init()

# Try different random seeds to avoid bad initialization
for seed in [0, 1, 2, 3, 4]:
    print(f"\n=== TRYING SEED {seed} ===")
    np.random.seed(seed)
    X, y = spiral_data(100, 3)
    
    # Initialize Layers with better architecture
    dense1 = Layer_Dense(2, 32)   # Much larger first layer
    activation1 = Activation_ReLU()
    
    dense2 = Layer_Dense(32, 64)  # Large second layer
    activation2 = Activation_ReLU()
    
    dense3 = Layer_Dense(64, 3)    # Output layer
    
    # Use combined softmax + loss for better stability
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
    
    # Training Parameters - try different learning rates
    best_accuracy = 0
    best_lr = 0
    
    for lr in [0.01, 0.05, 0.1, 0.5, 1.0]:
        print(f"  Testing learning rate: {lr}")
        
        # Reset network weights
        np.random.seed(seed)
        dense1 = Layer_Dense(2, 32)
        activation1 = Activation_ReLU()
        dense2 = Layer_Dense(32, 64)
        activation2 = Activation_ReLU()
        dense3 = Layer_Dense(64, 3)
        loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
        
        learning_rate = lr
        epochs = 5000
        
        for epoch in range(epochs):
            # Forward pass
            dense1_output = dense1.forward(X)
            activation1_output = activation1.forward(dense1_output)
            dense2_output = dense2.forward(activation1_output)
            activation2_output = activation2.forward(dense2_output)
            dense3_output = dense3.forward(activation2_output)
            
            # Calculate loss using combined function
            loss = loss_activation.forward(dense3_output, y)
            
            # Backpropagation
            loss_activation.backward(loss_activation.output, y)
            dense3.backward(loss_activation.dinputs)
            activation2.backward(dense3.dinputs)
            dense2.backward(activation2.dinputs)
            activation1.backward(dense2.dinputs)
            dense1.backward(activation1.dinputs)
            
            # Update weights
            dense1.weights -= learning_rate * dense1.dweights
            dense1.biases -= learning_rate * dense1.dbiases
            dense2.weights -= learning_rate * dense2.dweights
            dense2.biases -= learning_rate * dense2.dbiases
            dense3.weights -= learning_rate * dense3.dweights
            dense3.biases -= learning_rate * dense3.dbiases
            
            # Early stopping if we achieve good results
            if epoch % 1000 == 0:
                predictions = np.argmax(loss_activation.output, axis=1)
                accuracy = np.mean(predictions == y)
                if accuracy > 0.98:
                    break
        
        # Final evaluation
        predictions = np.argmax(loss_activation.output, axis=1)
        accuracy = np.mean(predictions == y)
        
        print(f"    Final - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_lr = lr
            best_predictions = predictions
            best_loss = loss
        
        # If we found excellent results, stop searching
        if accuracy > 0.98:
            print(f"    ✅ EXCELLENT RESULTS FOUND!")
            break
    
    print(f"  Best for seed {seed}: LR={best_lr}, Accuracy={best_accuracy:.4f}")
    
    # If we found great results with this seed, stop
    if best_accuracy > 0.95:
        print(f"\n🎉 SUCCESS with seed {seed}!")
        print(f"Final Loss: {best_loss:.4f}")
        print(f"Accuracy: {best_accuracy:.4f}")
        print(f"Predicted classes: {best_predictions[:10]}")
        print(f"True classes: {y[:10]}")
        print(f'Prediction distribution: {np.bincount(best_predictions)}')
        print(f'True distribution: {np.bincount(y)}')
        break

# If no seed worked well, provide debugging info
if best_accuracy < 0.9:
    print(f"\n⚠️ BEST RESULT ACROSS ALL SEEDS: {best_accuracy:.4f}")
    print("This suggests a deeper issue with the implementation.")
    print("Double-check gradient calculations and loss function.")