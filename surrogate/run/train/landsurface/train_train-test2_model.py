import pathlib as pl
import pandas as pd

from surrogate.run.train.utils.train_objective import train_objective

setup_input_dir = pl.Path("../../setup/landsurface/input")
setup_dir = pl.Path("../../setup/landsurface/saves/train-test2")
transform_dir = pl.Path("../../transform/landsurface/saves/train-test2")
out_dir = pl.Path("./saves/train-test2")
seed = 19920223
epochs = 90

final_learning_rate_fraction = 1e-1
learning_rate_loops = 3

params = {"sample_size": 32,
            "dates_size": 512,
            "n_lstm": 1,
            "n_in_linear": 1,
            "n_out_linear": 1,
            "in_hidden_size": 256,
            "out_hidden_size": 256,
            "dropout_rate": 0.15,
            "learning_rate": 1e-4,
            "transformer_type": "sqrt_standard"}

resolutions = [dir.stem for dir in setup_dir.iterdir() if dir.is_dir()]

resolution = resolutions[0]
for resolution in resolutions:
    print("Resolution: {}".format(resolution))
    
    setup_input_resolution_dir = pl.Path("{}/{}".format(setup_input_dir, resolution))
    setup_resolution_dir = pl.Path("{}/{}".format(setup_dir, resolution))
    transform_resolution_dir = pl.Path("{}/{}".format(transform_dir, resolution))
    out_resolution_dir = pl.Path("{}/{}".format(out_dir, resolution))

    in_features_file = pl.Path("{}/features_input.csv".format(setup_input_resolution_dir))
    in_features = pd.read_csv(in_features_file, keep_default_na=False).fillna("")
    out_features_file = pl.Path("{}/features_output.csv".format(setup_input_resolution_dir))
    out_features = pd.read_csv(out_features_file, keep_default_na=False).fillna("")

    samples_size = params["sample_size"]
    dates_size = params["dates_size"]
    n_lstm = params["n_lstm"]
    n_in_linear = params["n_in_linear"]
    n_out_linear = params["n_out_linear"]
    in_hidden_size = params["in_hidden_size"]
    out_hidden_size = params["out_hidden_size"]
    dropout_rate = params["dropout_rate"]
    learning_rate = params["learning_rate"]
    transformer_type = params["transformer_type"]
    
    transformer_type_dir = pl.Path("{}/{}".format(transform_resolution_dir, transformer_type))
    
    performance_out = pl.Path("{}/performance.csv".format(out_resolution_dir))
    state_dict_out = pl.Path("{}/state_dict.pt".format(out_resolution_dir))
    meta_out = pl.Path("{}/model_meta.pkl".format(out_resolution_dir))
    if performance_out.exists() and state_dict_out.exists() and meta_out.exists():
        print("Already exists")
        continue
    
    learning_rate_minimum = learning_rate * 1e-1
    learning_rate_maximum = learning_rate * 1e1
        
    train_dir = pl.Path("{}/train".format(setup_resolution_dir))
    validate_dir = pl.Path("{}/validate".format(setup_resolution_dir))
    test_dir = pl.Path("{}/test".format(setup_resolution_dir))
    
    trainer = train_objective(n_lstm=n_lstm,
                                n_in_linear=n_in_linear,
                                n_out_linear=n_out_linear,
                                in_hidden_size=in_hidden_size,
                                out_hidden_size=out_hidden_size,
                                dropout_rate=dropout_rate,
                                samples_size=samples_size,
                                dates_size=dates_size,
                                learning_rate=learning_rate_minimum,
                                learning_rate_maximum=learning_rate_maximum,
                                final_learning_rate_fraction=final_learning_rate_fraction,
                                learning_rate_loops=learning_rate_loops,
                                epochs=epochs,
                                train_dir=train_dir,
                                validation_dir=validate_dir,
                                test_dir=test_dir,
                                transformer_dir=transformer_type_dir,
                                out_dir=out_resolution_dir,
                                seed=seed,
                                input_features=in_features["feature"].to_numpy(),
                                output_features=out_features["feature"].to_numpy(),
                                verbose=2)
