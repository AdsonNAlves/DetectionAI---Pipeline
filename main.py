import argparse
from ultralytics import YOLO
from networks.structures import train_yolo, _read_yaml, _default_weights
import json
from pathlib import Path

def argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--project',
        help='Name of the project directory where training outputs are saved.',
        type=str,
        default='result_model')

    parser.add_argument(
        '--name',
        help='Name of the training run.',
        type=str,
        default='modelIASense')

    parser.add_argument(
        '--training',
        help='Training run.',
        type=bool,
        default=True)

    parser.add_argument(
        '--arch',
        help="Architecture to use ('v8' or 'v11')",
        type=str,
        default='v11')
    
    parser.add_argument(
    '--weights',
    help='Caminho para os pesos do modelo (checkpoint .pt)',
    type=str,
    default=None)

    return parser.parse_args()

def main(args=None):
    """
    Função principal para treinar um modelo usando os arquivos de configuração.
    args.yaml e data.yaml localizados em src/test/20250915v1.
    """
    # Caminhos para os arquivos de configuração
    test_dir = Path(__file__).parent / "test" / "20250915v1"
    data_yaml = test_dir / "data.yaml"
    args_yaml = test_dir / "args.yaml"

    arch = args.arch 

    print(f"Iniciando treinamento com arquitetura: {arch}")
    print(f"Usando data.yaml: {data_yaml}")
    print(f"Usando args.yaml: {args_yaml}")


    overrides = _read_yaml(args_yaml)
    overrides["project"] = args.project
    overrides["name"] = args.name
    overrides["data"] = str(data_yaml)
    if args.weights:
        overrides["model"] = args.weights
    else:
        overrides["model"] = _default_weights(args.arch)

    # Treina o modelo YOLO
    model = train_yolo(
        arch=arch,
        data_yaml=data_yaml,
        args_yaml=None,  # Não passa o caminho, passa o dict de overrides
        weights=None,
        overrides=overrides
    )

    print("Treinamento concluído. Último estado do modelo salvo.")

if __name__ == "__main__":
    args = argparser()
    main(args)