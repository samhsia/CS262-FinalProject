# CS262 Final Project: Noise-Tolerant Federated Learning

- To run the baseline experiment (i.e., no noise), open two terminals and run `cd src/`.
Run `python3 fl_server.py` in the first terminal and `python3 fl_client.py` in the second terminal.

- To enable noise injection, run `python3 fl_server.py` in the first terminal (unchanged) and `python3 fl_client.py --enable-malicious-agent "True" --noise-level 1 --num-malicious-agents 1` for the second.
Noise level and number of malicious agents can be adjusted with the `--noise-level` and `num-malicious-agents` flags, respectively.

- Lastly, to enable anomaly detection and recovery, run `python3 fl_server.py --enable-anomaly-detection "True"` for the first terminal and `python3 fl_client.py --enable-malicious-agent "True" --noise-level 1 --num-malicious-agents 1` for the second.
Again, noise level and number of malicious agents can be adjusted with the `--noise-level` and `num-malicious-agents` flags, respectively.

Our design notebook is located [here](https://docs.google.com/document/d/1jnjmIV0m2-V_2n1Y7F1b-T_4Ihv0LchmqN-aC5gg7Z8/edit?usp=sharing).
