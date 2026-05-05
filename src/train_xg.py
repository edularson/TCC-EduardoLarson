from modules.xg_model import XGModel
from sklearn.model_selection import train_test_split

xg = XGModel()
X, y = xg.load_statsbomb_data('data/statsbomb_shots_laliga.csv')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xg.train(X_train, y_train)

metrics = xg.evaluate(X_test, y_test)
print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
print(f"Log Loss: {metrics['log_loss']:.4f}")
print(f"Brier Score: {metrics['brier_score']:.4f}")

xg.save_model()