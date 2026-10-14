"""Train the residual backbone through the common patient-separated protocol."""
from ssrl_ecg.cli import run_training_cli

def main():
    run_training_cli('supervised', backbone='resnet')

if __name__ == '__main__':
    main()
