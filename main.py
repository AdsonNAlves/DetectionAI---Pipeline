import argparse
from pathlib import Path
# Usa o dispatcher com suporte a YOLO v8/v11 e Faster R-CNN
from networks.structures import train_model, _read_yaml, _default_weights

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
        help="Architecture to use: 'v8', 'v11' or 'frcnn'",
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


    # 1) Carrega args.yaml base
    overrides = _read_yaml(args_yaml)
    # 2) CLI overrides (CLI sempre sobrescreve YAML)
    overrides["project"] = args.project
    overrides["name"] = args.name
    overrides["data"] = str(data_yaml)
    # Mantém a chave 'model' para compatibilidade com fluxos antigos
    if args.weights:
        overrides["model"] = args.weights
    else:
        # Para YOLO, define um default; para FRCNN, será ignorado
        overrides["model"] = _default_weights(args.arch) if args.arch.lower() in {"v8","yolov8","yolo8","v11","yolov11","yolo11"} else None

    # # Decide pesos a passar explicitamente (apenas YOLO usa)
    # weights_to_pass = args.weights if args.arch.lower() in {"v8","yolov8","yolo8","v11","yolov11","yolo11"} else None

    # Treina o modelo conforme a arquitetura escolhida
    model = train_model(
        arch=arch,
        data_yaml=data_yaml,
        args_yaml=None,  # usamos o dict overrides diretamente
        weights=None,
        overrides=overrides,
    )

    print("Treinamento concluído. Último estado do modelo salvo.")

if __name__ == "__main__":
    args = argparser()
    main(args)
