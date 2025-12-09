python ./run_single_simulation_buyer_communication.py r_wbc_siloed.db reputation_only siloed

python ./run_single_simulation_buyer_communication.py r_wbc_shared.db reputation_only shared

python ./run_single_simulation_buyer_communication.py rw_wbc_siloed.db reputation_and_warrant siloed

python ./run_single_simulation_buyer_communication.py rw_wbc_shared.db reputation_and_warrant shared

# TODO need support

# python ./run_experiments.py --experiment-id r_wbc_siloed --type buyer_communication --market-type reputation_only --conditions siloed

# python ./run_experiments.py --experiment-id r_wbc_shared --type buyer_communication --market-type reputation_only --conditions shared

# python ./run_experiments.py --experiment-id rw_wbc --type buyer_communication --market-type reputation_and_warrant --conditions siloed

# python ./run_experiments.py --experiment-id rw_wbc --type buyer_communication --market-type reputation_and_warrant --conditions shared