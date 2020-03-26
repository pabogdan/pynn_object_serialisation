"""
Utility script to retrieve the HASH of the current git commit

Running command in bash:
https://stackoverflow.com/questions/4256107/running-bash-commands-in-python

Retrieving current commit HASH:
https://stackoverflow.com/questions/949314/how-to-retrieve-the-hash-for-the-current-commit-in-git
"""
import numpy as np
# Making the generator for the images
from keras_rewiring.utilities.imagenet_utils import ImagenetDataGenerator

def retrieve_git_commit():
    import subprocess
    from subprocess import PIPE
    bash_command = "git rev-parse HEAD"

    try:
        # We have to use `stdout=PIPE, stderr=PIPE` instead of `text=True`
        # when using Python 3.6 and earlier. Python 3.7+ will have these QOL
        # improvements
        proc = subprocess.run(bash_command.split(),
                              stdout=PIPE, stderr=PIPE, shell=False)
        return proc.stdout
    except subprocess.CalledProcessError as e:
        print("Failed to retrieve git commit HASH-", str(e))
        return "CalledProcessError"
    except Exception as e:
        print("Failed to retrieve git commit HASH more seriously-", str(e))
        return "GeneralError"


def compute_input_spikes(no_tests, data_path, input_size, t_stim, rate_scaling):
    image_length = int(np.sqrt((input_size / 3)))
    image_size = (image_length, image_length, 3)
    
    print("=" * 80)
    print("Experiment mini-report")
    print("-" * 80)
    print("Number of testing examples to use:", no_tests)

    generator = ImagenetDataGenerator('val', no_tests, data_path, image_size)
    gen = generator()
    x_test, y_test = gen.__next__()
    # reshape input to flatten data
    # x_train = x_train.reshape(x_train.shape[0], np.prod(x_train.shape[1:]))
    x_test = x_test.reshape(x_test.shape[0], np.prod(x_test.shape[1:]))

    runtime = no_tests * t_stim
    number_of_slots = int(runtime / t_stim)
    range_of_slots = np.arange(number_of_slots)
    starts = np.ones((input_size, number_of_slots)) * (range_of_slots * t_stim)
    durations = np.ones((input_size, number_of_slots)) * t_stim
    rates = x_test[:no_tests].T

    # scaling rates
    print("=" * 80)
    print("Scaling rates...")
    min_rates = np.min(rates)
    max_rates = np.max(rates)
    _0_to_1_rates = rates - min_rates
    print("rates - min_rates min", np.min(_0_to_1_rates))
    print("rates - min_rates max", np.max(_0_to_1_rates))
    _0_to_1_rates = _0_to_1_rates / float(np.max(_0_to_1_rates))
    print("_0_to_1_rates min", np.min(_0_to_1_rates))
    print("_0_to_1_rates max", np.max(_0_to_1_rates))
    rates = _0_to_1_rates * rate_scaling

    print("Finished scaling rates...")
    print("=" * 80)
    # Let's do some reporting about here
    print("Going to put in", no_tests, "images")
    print("The shape of the rates array is ", rates.shape)
    print("This shape is supposed to match that of durations ", durations.shape)
    print("... and starts ", starts.shape)

    assert (rates.shape == durations.shape)
    assert (rates.shape == starts.shape)
    assert (rates.shape[1] == no_tests)

    print("Input image size is expected to be ", image_size)
    print("... i.e. ", np.prod(image_size), "pixels")
    print("Mobilenet generally expects the image size to be ", (224, 224, 3))

    print("Min rate", np.min(rates))
    print("Max rate", np.max(rates))
    print("Mean rate", np.mean(rates))
    print("=" * 80)

    # return input_params
    input_params = {
        "rates": rates,
        "durations": durations,
        "starts": starts
    }
    return input_params, y_test