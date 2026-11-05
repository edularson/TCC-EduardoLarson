"""Processa um clipe com o pipeline do TCC, sem executar sua avaliação histórica."""

import argparse
import logging
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True, help='Clipe a processar')
    parser.add_argument('--output', type=Path, required=True, help='Vídeo de saída .mp4')
    parser.add_argument('--team-video', type=Path, help='Trecho da mesma partida para ajustar os times')
    parser.add_argument('--config', type=Path, help='Configuração YAML alternativa')
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    if Path.cwd().resolve() != root:
        parser.error('Execute a partir da raiz do repositório.')
    for label, path in [('vídeo', args.video), ('vídeo de times', args.team_video), ('configuração', args.config)]:
        if path is not None and not path.is_file():
            parser.error(f'Arquivo de {label} não encontrado: {path}')
    if args.output.suffix.lower() != '.mp4':
        parser.error('A saída deve usar a extensão .mp4.')
    if args.output.resolve() in {args.video.resolve(), (args.team_video or args.video).resolve()}:
        parser.error('A saída deve ser diferente dos vídeos de entrada.')

    sys.path.insert(0, str(root / 'src'))
    # Carrega dependências e modelos somente após validar os argumentos.
    import yaml

    config_path = args.config or root / 'src/configs/config.yaml'
    with config_path.open(encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    models = config['models']
    required = [
        models['detection']['path'],
        models['ball_detection']['path'],
        models['field_detection']['weights_kp'],
        models['field_detection']['weights_line'],
        'runs/xg_model.joblib',
        'external/PnLCalib/config/hrnetv2_w48.yaml',
        'external/PnLCalib/config/hrnetv2_w48_l.yaml',
    ]
    missing = [str(path) for path in required if not Path(path).is_file()]
    if missing:
        parser.error('Prepare os arquivos descritos em docs/reproducibilidade.md:\n' + '\n'.join(missing))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    (root / 'outputs').mkdir(exist_ok=True)
    logging.basicConfig(filename=root / 'outputs/process_video.log', level=logging.INFO)

    from main import VarzeaVisionPipeline

    pipeline = VarzeaVisionPipeline(str(config_path))
    pipeline.auto_train_team_classifier(str(args.team_video or args.video))
    results = pipeline.process_clip(str(args.video), output_path=str(args.output))
    print(f'Processamento concluído: {args.output}')
    print(f'Finalizações detectadas: {len(results["shots"])}')
    if results.get('possession'):
        print(f'Posse estimada: {results["possession"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
