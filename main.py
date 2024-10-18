import src.utils as utils


def main():
    """
    Main function that loads a config file and reads arguments from the command line.

    This function reads command line arguments and then loads the configuration from a YAML file.
    It then sets up the model for the appropriate dataset and then masks the sentences before
    denoising them. Finally, it does the prediction.
    """
    # Set up seed
    utils.setup_seed()

    # Set up command-line argument parsing
    args = utils.parse_args()

    # Load configuration from YAML file
    config = utils.load_config(args.config)

    # Either certify or attack
    if args.mode == "certify":
        utils.certify(args, config)
    elif args.mode == "attack":
        utils.attack(args, config)


if __name__ == "__main__":
    main()
