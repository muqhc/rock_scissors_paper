import numpy as np
import tensorflow as tf
import keras
import os
import matplotlib.pyplot as plt
from keras import losses, backend, models, layers

model_name = "my_model"

LOAD_EXIST = os.path.exists(f"{model_name}.keras")
TRAINING = True

state_count = 4

if not LOAD_EXIST:
    # Define the model
    model = models.Sequential([
        layers.Dense(20, input_dim=3+state_count, activation='relu'),  # Hidden layer
        layers.Dense(3+state_count, activation='sigmoid')             # Output layer
    ])

    # Compile the model
    model.compile(optimizer='SGD', loss='mean_squared_error')

if LOAD_EXIST:
    model = keras.models.load_model("my_model.keras")

lossf: losses.Loss = losses.MeanSquaredError()
optimizer: keras.optimizers.Optimizer = model.optimizer

# Simulate training data
M = 1
X_train = np.zeros((M, 3+state_count))  # 3 samples, 10 features
y_train = np.zeros((M, 3+state_count))  # 3 samples, 10 target values

# Parameters for recursive feedback
num_iterations = 8  # Number of recursive steps
epochs = 30         # Training epochs
batch_size = 16
noise_scale = 0.006

# region: utils
def encode_rcp_win(move: str):
    if move == "a": return np.array([0,1,0])
    if move == "s": return np.array([0,0,1])
    if move == "d": return np.array([1,0,0])
    return encode_rcp_win(input("re-move(asd): "))

def prediction2matrix(pred: np.ndarray):
    return np.array([
        [pred[0],pred[1],pred[2]],
        [pred[2],pred[0],pred[1]],
        [pred[1],pred[2],pred[0]],
        ])

def print_compare(x):
    if (x>0.5): print("AI Win")
    elif (x<-0.5): print("AI Lose")
    else: print("Draw")
    return x

def compare_move(ai: np.ndarray, win_player: np.ndarray):
    return ((ai.argmax()-((win_player.argmax()-1)%3)+1)%3-1)

def encode_compared(compared: float):
    return np.array([1,0,0]) if compared > 0.5 else np.array([0,0,1]) if compared < -0.5 else np.array([0,1,0])

def y_from_move(past_win: np.ndarray, win_player: np.ndarray):
    trace = (past_win.argmax() - win_player.argmax()) % 3
    if (trace == 0): return np.array([1,0,0])
    elif (trace == 1): return np.array([0,1,0])
    else: return np.array([0,0,1])

# endregion

total_score_container: list[float] = []

# Custom training loop
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    batch_X = X_train.copy()
    batch_y = y_train.copy()

    noise = lambda: np.concat([np.zeros((M,3)),(np.random.normal(0,noise_scale,(M, state_count)))],axis=1)

    # Initialize input for recursive chain
    current_input = batch_X + noise()
    past_player_win_move = encode_rcp_win(input("move(asd): "))
    score = np.array([encode_compared(print_compare(np.random.choice(np.array([-1.0,0.0,1.0])))) for i in range(M)])
    current_input[:,:3] = score[:,:3]
    
    total_score = 0

    # Recursive chaining
    for t in range(num_iterations):
        print("------")
        # Forward pass
        with tf.GradientTape() as tape:
            predictions = model(current_input, training=TRAINING)
            for n in range(3,state_count+3):
                for m in range(M):
                    batch_y[m,n] = predictions[m,n]
            for m in range(M):
                mat = prediction2matrix(predictions[m,:3])
                true_pred = mat @ past_player_win_move
                win_move_against_player = encode_rcp_win(input("move(asd): "))
                pred_oriented_y = y_from_move(past_player_win_move, win_move_against_player)
                # print(np.argmax(true_pred) == np.argmax(win_move_against_player))
                score_raw = compare_move(true_pred, win_move_against_player)
                total_score += score_raw*(1.5**t)
                print_compare(score_raw)
                print(score_raw, true_pred, win_move_against_player)
                score[m,:] = encode_compared(score_raw)
                for n in range(3):
                    batch_y[m,n] = pred_oriented_y[n]
                past_player_win_move = win_move_against_player
                
            loss = lossf.call(batch_y, predictions)*state_count
        
        # Backward pass (gradient calculation and application)
        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.learning_rate = 0.08*np.tanh(t/2+0.1)
        if TRAINING: optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        
        # Use output as the next input
        current_input = np.concat([score, predictions[:,3:state_count+3]], axis=1)
    
    print(f"Epoch {epoch}: Loss = ", loss)
    print(total_score)
    total_score_container.append(total_score)


model.save(f"{model_name}.keras")

print("Total score:", total_score)

fig, axs = plt.subplots(1,1)
axs.set(ylim=(-50,50))
axs.bar(np.arange(epochs), np.array(total_score_container), color=["red" if s < 0 else "green" for s in total_score_container])
axs.set_ylabel("Score")
axs.set_xlabel("epoch")
fig.savefig("Figure.png")