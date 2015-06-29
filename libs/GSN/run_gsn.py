'''
This scripts produces the model trained on MNIST discussed in the paper:

'Deep Generative Stochastic Networks Trainable by Backprop'
Yoshua Bengio, Eric Thibodeau-Laufer
http://arxiv.org/abs/1306.1091
'''
import argparse
import model

import numpy as np

def main():
    parser = argparse.ArgumentParser()
    # Add options here

    parser.add_argument('--K', type=int, default=2) # nubmer of hidden layers
    parser.add_argument('--N', type=int, default=4) # number of walkbacks
    parser.add_argument('--n_epoch', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--hidden_add_noise_sigma', type=float, default=2)
    parser.add_argument('--input_salt_and_pepper', type=float, default=0.4)
    parser.add_argument('--learning_rate', type=float, default=0.25)
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--annealing', type=float, default=0.995)
    parser.add_argument('--hidden_size', type=float, default=1500)
    parser.add_argument('--act', type=str, default='tanh')
    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--data_path', type=str, default='.')
   
    # argparse does not deal with bool 
    parser.add_argument('--vis_init', type=int, default=0)
    parser.add_argument('--noiseless_h1', type=int, default=1)
    parser.add_argument('--input_sampling', type=int, default=1)
    parser.add_argument('--test_model', type=int, default=0)
    
    args = parser.parse_args()
   
    print args.test_model 
    
    from sklearn.datasets import fetch_lfw_people
    def build_lfw():
        lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.2)
        X = lfw_people.data
        X = X.astype(np.float32) / 255.
        return X
    X = build_lfw()
    y = None

    model.experiment(args, None, ((X, y), (X, y), (X, y)) )
    
if __name__ == '__main__':
    main()
