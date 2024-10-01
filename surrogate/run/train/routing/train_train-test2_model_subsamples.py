import pathlib as pl
import pandas as pd

from surrogate.run.train.utils.train_objective import train_objective

setup_input_dir = pl.Path("../../setup/routing/input")
setup_dir = pl.Path("../../setup/routing/saves/train-test2")
transform_dir = pl.Path("../../transform/routing/saves/train-test2")
out_dir = pl.Path("./subsamples/train-test2")
seed = 19920223

resolution = "mulres"
subsamples = [2, 4, 8, 16, 32]
epochs = 90

final_learning_rate_fraction = 1e-1
learning_rate_loops = 3

params = {"lake": {"sample_size": 32,
                   "dates_size": 768,
                   "n_lstm": 1,
                   "n_in_linear": 1,
                   "n_out_linear": 1,
                   "in_hidden_size": 256,
                   "out_hidden_size": 256,
                   "dropout_rate": 0.2,
                   "learning_rate": 5e-5,
                   "transformer_type": "losg-1p0_standard"},
          "reservoir": {"sample_size": 32,
                        "dates_size": 768,
                        "n_lstm": 1,
                        "n_in_linear": 1,
                        "n_out_linear": 1,
                        "in_hidden_size": 256,
                        "out_hidden_size": 256,
                        "dropout_rate": 0.7,
                        "learning_rate": 1e-5,
                        "transformer_type": "log_standard"},
          "river": {"sample_size": 32,
                    "dates_size": 768,
                    "n_lstm": 1,
                    "n_in_linear": 1,
                    "n_out_linear": 1,
                    "in_hidden_size": 256,
                    "out_hidden_size": 256,
                    "dropout_rate": 0.15,
                    "learning_rate": 5e-5,
                    "transformer_type": "log_standard"}}

setup_input_resolution_dir = pl.Path("{}/{}".format(setup_input_dir, resolution))
setup_resolution_dir = pl.Path("{}/{}".format(setup_dir, resolution))
transform_resolution_dir = pl.Path("{}/{}".format(transform_dir, resolution))
out_resolution_dir = pl.Path("{}/{}".format(out_dir, resolution))

in_features_file = pl.Path("{}/features_input.csv".format(setup_input_resolution_dir))
in_features = pd.read_csv(in_features_file, keep_default_na=False).fillna("")
out_features_file = pl.Path("{}/features_output.csv".format(setup_input_resolution_dir))
out_features = pd.read_csv(out_features_file, keep_default_na=False).fillna("")

routing_types = [dir.stem for dir in setup_resolution_dir.iterdir() if dir.is_dir()]

routing_type = routing_types[0]
for routing_type in routing_types:
    print("Routing type: {}".format(routing_type))
            
    setup_routing_dir = pl.Path("{}/{}".format(setup_resolution_dir, routing_type))
    transform_routing_dir = pl.Path("{}/{}".format(transform_resolution_dir, routing_type))
    out_routing_dir = pl.Path("{}/{}".format(out_resolution_dir, routing_type))

    samples_size = params[routing_type]["sample_size"]
    dates_size = params[routing_type]["dates_size"]
    n_lstm = params[routing_type]["n_lstm"]
    n_in_linear = params[routing_type]["n_in_linear"]
    n_out_linear = params[routing_type]["n_out_linear"]
    in_hidden_size = params[routing_type]["in_hidden_size"]
    out_hidden_size = params[routing_type]["out_hidden_size"]
    dropout_rate = params[routing_type]["dropout_rate"]
    learning_rate = params[routing_type]["learning_rate"]
    transformer_type = params[routing_type]["transformer_type"]
    
    transformer_type_dir = pl.Path("{}/{}".format(transform_routing_dir, transformer_type))
    
    learning_rate_minimum = learning_rate * 1e-1
    learning_rate_maximum = learning_rate * 1e1
        
    train_dir = pl.Path("{}/train".format(setup_routing_dir))
    validate_dir = pl.Path("{}/validate".format(setup_routing_dir))
    test_dir = pl.Path("{}/test".format(setup_routing_dir))
    
    for subsample in subsamples:
        print("\tSubsample: {}".format(subsample))
        
        out_subsample_dir = pl.Path("{}/{}".format(out_routing_dir, subsample))
    
        performance_out = pl.Path("{}/performance.csv".format(out_subsample_dir))
        state_dict_out = pl.Path("{}/state_dict.pt".format(out_subsample_dir))
        meta_out = pl.Path("{}/model_meta.pkl".format(out_subsample_dir))
        if performance_out.exists() and state_dict_out.exists() and meta_out.exists():
            print("Already exists")
            continue
        
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
                                    train_split_fraction=1 / subsample,
                                    validation_split_fraction=1 / subsample,
                                    test_split_fraction=1 / subsample,
                                    transformer_dir=transformer_type_dir,
                                    out_dir=out_subsample_dir,
                                    seed=seed,
                                    input_features=in_features["feature"].to_numpy(),
                                    output_features=out_features["feature"].to_numpy(),
                                    verbose=2)
