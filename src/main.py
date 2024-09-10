import argparse
from energy_data import prepare_data
from train import train_model
from inference import run_inference

def main():
    parser = argparse.ArgumentParser(description="Energy Prediction Pipeline")
    parser.add_argument('phase', choices=['prepare', 'train', 'inference'], help="Phase to run: prepare, train, inference")
    args = parser.parse_args()

    if args.phase == 'prepare':
        prepare_data()
    elif args.phase == 'train':
        train_model()
    elif args.phase == 'inference':
        run_inference()

if __name__ == '__main__':
    main()
