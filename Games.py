import numpy as np

# Features: hours spent (Football, Tennis, Cricket)
X1 = 5   # Football hours
X2 = 1   # Tennis hours
X3 = 2   # Cricket hours

# Coefficients for log-odds equations
# Football vs Cricket: logit_F = -1 + 0.7*X1 - 0.2*X2 - 0.1*X3
# Tennis vs Cricket:   logit_T = 0.5 - 0.3*X1 + 0.8*X2 - 0.2*X3

logit_F = -1 + 0.7*X1 - 0.2*X2 - 0.1*X3
logit_T = 0.5 - 0.3*X1 + 0.8*X2 - 0.2*X3

# For Cricket (baseline) -> logit = 0
logits = [0, logit_F, logit_T]

# Exponentiate
exp_logits = np.exp(logits)

# Softmax probabilities
probs = exp_logits / np.sum(exp_logits)

# Display results
sports = ["Cricket ", "Football", "Tennis"]
for s, p in zip(sports, probs):
    print(f"P({s}) = {p:.2f}")

# Prediction
predicted = sports[np.argmax(probs)]
print("\nFinal Prediction:", predicted)


