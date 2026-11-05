# Preparação e reprodução

## Estado desta publicação

O repositório contém o código fornecido pelo autor, a configuração, o texto final, os slides, exemplos visuais e um clipe processado de demonstração em `docs/videos/demo-arg-fra.mp4`, com prévia animada no README. Os conjuntos de vídeos de avaliação, datasets, ambientes virtuais, pesos treinados e bibliotecas externas não foram copiados. A instalação e a inferência completas ainda precisam ser validadas em um ambiente limpo.

O `requirements.txt` da raiz preserva o ambiente exportado pelo autor. O arquivo `requirements-research.txt` preserva uma segunda lista encontrada no projeto; as duas listas divergem em algumas versões e não devem ser instaladas em sequência. A referência de Python indicada nos materiais locais é 3.12.

## Componentes externos

A integração em `src/modules/field_detection.py` espera os diretórios abaixo na raiz:

```text
external/PnLCalib/
external/no_bells_just_whistles/
```

Referências de origem registradas nos READMEs locais: [PnLCalib](https://github.com/mguti97/PnLCalib) e [No Bells Just Whistles](https://github.com/mguti97/No-Bells-Just-Whistles). Prepare essas dependências e suas dependências Python conforme os projetos originais. As revisões exatas das cópias locais não estão registradas nesta publicação.

O pacote `sports` está fixado por commit no `requirements.txt` e é instalado diretamente do [Roboflow Sports](https://github.com/roboflow/sports).

## Pesos e arquivos necessários

| Componente | Caminho padrão na configuração |
| --- | --- |
| Detector geral | `src/runs/detect/train/weights/best.pt` |
| Detector de bola | `src/runs/detect/ball_only/weights/best.pt` |
| Pontos-chave do campo | `external/PnLCalib/weights/SV_FT_WC14_kp.pth` |
| Linhas do campo | `external/PnLCalib/weights/SV_FT_TSWC_lines.pth` |
| Modelo xG treinado | `runs/xg_model.joblib` |

Os caminhos dos quatro primeiros itens estão em `src/configs/config.yaml`. Os pesos do detector geral e de bola treinados pelo autor ainda não têm um endereço de download publicado aqui. Os pesos de campo devem ser obtidos na origem do PnLCalib, respeitando suas condições e usando nomes compatíveis com a configuração.

Para treinar o modelo de xG com as competições configuradas:

```bash
python src/tools/train_xg.py
```

O script consulta o StatsBomb Open Data e pode levar tempo. O protocolo atual de calibração tem a limitação descrita em [resultados](resultados.md); repetir esse treinamento não corrige essa limitação.

## Processar um vídeo

Depois de preparar o ambiente, componentes externos e arquivos:

```bash
python scripts/process_video.py --video caminho/clipe.mp4 --output outputs/clipe.mp4
```

Opcionalmente use `--team-video caminho/trecho_uniformes.mp4` para ajustar o classificador de times em outro trecho da mesma partida, ou `--config caminho/config.yaml` para outra configuração. Execute a partir da raiz do repositório. Os clipes devem ser fornecidos pelo usuário conforme suas permissões de uso.

O script cria o diretório de saída e configura o log antes de inicializar o pipeline. O uso direto de `src/main.py` mantém o roteiro específico do experimento original e não é a entrada recomendada.

## Ajustes desta publicação

- Foi acrescentado um comando parametrizado em `scripts/process_video.py`, sem os rótulos de avaliação do roteiro histórico.
- Foi corrigida a importação de `get_project_root` em `team_classifier.py`, que já usava essa função.
- Foram removidos espaços ao final de linhas para padronizar a formatação.
- Os algoritmos e o treinamento fornecidos foram preservados; não houve retreinamento ou recálculo de métricas.

## Verificação realizada

Foi verificada a sintaxe dos arquivos Python e os links relativos da documentação. A ajuda do comando e a rejeição de vídeo inexistente foram verificadas sem carregar modelos. Esses checks não equivalem a uma execução de inferência.
