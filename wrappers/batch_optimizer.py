from hp_toolkit.hp import Param, default_eval_functions

params = dict(
    batch_size = Param(initial=128, interval=[10, 50, 100, 128, 256, 512], type='choice'),
    learning_rate = Param(initial=0.01, interval=[-5, -3], type='real', scale='log10'),
    momentum = Param(initial=0.5, interval=[0.5, 0.8, 0.9, 0.95, 0.99], type='choice'),
    max_epochs = Param(initial=100, interval=[200], type='choice')
)
