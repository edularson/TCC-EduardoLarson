import yaml
import copy
from pathlib import Path
from functools import lru_cache


@lru_cache(maxsize=4)
def _load_config_cached(config_path: str) -> dict:
    """
    Carrega e cacheia o YAML em disco.
    Separado de load_config para que o cache use sempre uma chave string
    normalizada — evita carregar duas vezes quando um módulo passa None
    e outro passa o path explícito.

    ATENÇÃO: retorna o dict original (não copiado) apenas aqui.
    Cópias defensivas são feitas em load_config().
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_config(config_path=None) -> dict:
    """
    Carrega config.yaml com resolução automática de path.

    C11 — dois problemas corrigidos em relação à versão anterior:

    1. lru_cache com argumento None vs path explícito criava duas entradas
       de cache distintas, fazendo o arquivo ser lido duas vezes e o cache
       (maxsize=1) descartar uma delas a cada chamada alternada.
       Solução: normaliza config_path para string absoluta ANTES de passar
       ao cache — todas as chamadas que resolvem para o mesmo arquivo
       compartilham a mesma entrada de cache.

    2. O dict cacheado era o MESMO objeto em memória retornado a todos os
       módulos. Qualquer módulo que fizesse self.config['x'] = y corromperia
       o config de todos os outros silenciosamente.
       Solução: retorna sempre uma cópia rasa (copy.copy). Para a maioria
       dos usos (leitura de valores escalares e sub-dicts não modificados)
       isso é suficiente. Se um módulo modificar um sub-dict, use copy.deepcopy.
    """
    if config_path is None:
        module_dir   = Path(__file__).parent
        project_root = module_dir.parent
        config_path  = project_root / "configs" / "config.yaml"

    # Normaliza para string absoluta — chave de cache consistente
    resolved = str(Path(config_path).resolve())

    raw = _load_config_cached(resolved)

    # Cópia defensiva: protege o cache de mutação acidental
    return copy.copy(raw)


def get_project_root() -> Path:
    """Retorna path absoluto para a raiz do projeto."""
    return Path(__file__).parent.parent


if __name__ == "__main__":
    cfg = load_config()
    cfg2 = load_config()
    assert cfg is not cfg2, "load_config deve retornar objetos distintos"
    print("Module test passed. common.py ready.")
    print(f"Project root: {get_project_root()}")