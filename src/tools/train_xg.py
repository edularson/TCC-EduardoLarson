import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modules.xg_model import XGModel

def main():
    print("Iniciando pipeline de treinamento do modelo xG...")
    
    xg = XGModel()

    metrics = xg.train()


    print("\n" + "="*40)
    print("RESUMO DO TREINAMENTO")
    print("="*40)
    print(f"Total de Chutes: {metrics['n_shots']}")
    print(f"AUC-ROC:         {metrics['auc']:.4f}")
    print(f"Brier Score:     {metrics['brier']:.4f}")
    print(f"xG Médio:        {metrics['mean_xg']:.4f}")

if __name__ == "__main__":
    main()