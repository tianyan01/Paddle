paddle.fluid.optimizer.PipelineOptimizer (paddle.fluid.optimizer.PipelineOptimizer, ('document', '2e55a29dbeb874934f7a1a1af3a22b8c'))
paddle.fluid.optimizer.PipelineOptimizer.__init__ (ArgSpec(args=['self', 'optimizer', 'num_microbatches', 'start_cpu_core_id'], varargs=None, keywords=None, defaults=(1, 0)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))
paddle.fluid.optimizer.PipelineOptimizer.minimize (ArgSpec(args=['self', 'loss', 'startup_program', 'parameter_list', 'no_grad_set'], varargs=None, keywords=None, defaults=(None, None, None)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))
paddle.audio.features (ArgSpec(), ('document', 'd41d8cd98f00b204e9800998ecf8427e'))
paddle.audio.features.layers.LogMelSpectrogram (ArgSpec(), ('document', 'c38b53606aa89215c4f00d3833e158b8'))
paddle.audio.features.layers.LogMelSpectrogram.forward (ArgSpec(args=['self', 'x'], varargs=None, varkw=None, defaults=None, kwonlyargs=[], kwonlydefaults=None, annotations={'return': <class 'paddle.Tensor'>, 'x': <class 'paddle.Tensor'>}), ('document', '6c14f6f78dc697a6981cf90412e2f1ea'))
paddle.audio.features.layers.LogMelSpectrogram.load_dict (ArgSpec(args=[], varargs='args', varkw='kwargs', defaults=None, kwonlyargs=[], kwonlydefaults=None, annotations={}), ('document', '01221a60445ee437f439a8cbe293f759'))
paddle.audio.features.layers.LogMelSpectrogram.state_dict (ArgSpec(args=['self', 'destination', 'include_sublayers', 'structured_name_prefix', 'use_hook'], varargs=None, varkw=None, defaults=(None, True, '', True), kwonlyargs=[], kwonlydefaults=None, annotations={}), ('document', '0c01cb0c12220c9426ae49549b145b0b'))
paddle.audio.features.layers.MFCC (ArgSpec(), ('document', 'bcbe6499830d9228a4f746ddd63b6c0f'))
paddle.audio.features.layers.MFCC.forward (ArgSpec(args=['self', 'x'], varargs=None, varkw=None, defaults=None, kwonlyargs=[], kwonlydefaults=None, annotations={'return': <class 'paddle.Tensor'>, 'x': <class 'paddle.Tensor'>}), ('document', 'd86bcaa345f26851089bfdb3efecd9e7'))
paddle.audio.features.layers.MelSpectrogram (ArgSpec(), ('document', 'adf4012310984568ae9da6170aa89f91'))
paddle.audio.features.layers.MelSpectrogram.forward (ArgSpec(args=['self', 'x'], varargs=None, varkw=None, defaults=None, kwonlyargs=[], kwonlydefaults=None, annotations={'return': <class 'paddle.Tensor'>, 'x': <class 'paddle.Tensor'>}), ('document', '458e9d454c8773091567c6b400f48cf5'))
paddle.audio.features.layers.Spectrogram (ArgSpec(), ('document', '83811af6da032099bf147e3e01a458e1'))
paddle.audio.features.layers.Spectrogram.forward (ArgSpec(args=['self', 'x'], varargs=None, varkw=None, defaults=None, kwonlyargs=[], kwonlydefaults=None, annotations={'return': <class 'paddle.Tensor'>, 'x': <class 'paddle.Tensor'>}), ('document', 'ab11e318fca1410f743b5432394dea35'))
paddle.audio.functional (ArgSpec(), ('document', 'd41d8cd98f00b204e9800998ecf8427e'))
paddle.audio.functional.functional.compute_fbank_matrix (ArgSpec(args=['sr', 'n_fft', 'n_mels', 'f_min', 'f_max', 'htk', 'norm', 'dtype'], varargs=None, varkw=None, defaults=(64, 0.0, None, False, 'slaney', 'float32'), kwonlyargs=[], kwonlydefaults=None, annotations={'return': <class 'paddle.Tensor'>, 'sr': <class 'int'>, 'n_fft': <class 'int'>, 'n_mels': <class 'int'>, 'f_min': <class 'float'>, 'f_max': typing.Union[float, NoneType], 'htk': <class 'bool'>, 'norm': typing.Union[str, float], 'dtype': <class 'str'>}), ('document', '3c5411caa6baedb68860b09c81e0147c'))
paddle.audio.functional.functional.create_dct (ArgSpec(args=['n_mfcc', 'n_mels', 'norm', 'dtype'], varargs=None, varkw=None, defaults=('ortho', 'float32'), kwonlyargs=[], kwonlydefaults=None, annotations={'return': <class 'paddle.Tensor'>, 'n_mfcc': <class 'int'>, 'n_mels': <class 'int'>, 'norm': typing.Union[str, NoneType], 'dtype': <class 'str'>}), ('document', 'c9c57550671f9725b053769411d2f65a'))
paddle.audio.functional.functional.fft_frequencies (ArgSpec(args=['sr', 'n_fft', 'dtype'], varargs=None, varkw=None, defaults=('float32',), kwonlyargs=[], kwonlydefaults=None, annotations={'return': <class 'paddle.Tensor'>, 'sr': <class 'int'>, 'n_fft': <class 'int'>, 'dtype': <class 'str'>}), ('document', '057b990e79c9c780622407267c0a43c6'))
paddle.audio.functional.functional.hz_to_mel (ArgSpec(args=['freq', 'htk'], varargs=None, varkw=None, defaults=(False,), kwonlyargs=[], kwonlydefaults=None, annotations={'return': typing.Union[paddle.Tensor, float], 'freq': typing.Union[paddle.Tensor, float], 'htk': <class 'bool'>}), ('document', '7ca01521dd0bf26cd3f72c67f7168dc4'))
paddle.audio.functional.functional.mel_frequencies (ArgSpec(args=['n_mels', 'f_min', 'f_max', 'htk', 'dtype'], varargs=None, varkw=None, defaults=(64, 0.0, 11025.0, False, 'float32'), kwonlyargs=[], kwonlydefaults=None, annotations={'return': <class 'paddle.Tensor'>, 'n_mels': <class 'int'>, 'f_min': <class 'float'>, 'f_max': <class 'float'>, 'htk': <class 'bool'>, 'dtype': <class 'str'>}), ('document', '2af3cf997ed1274214ec240b2b59a98d'))
paddle.audio.functional.functional.mel_to_hz (ArgSpec(args=['mel', 'htk'], varargs=None, varkw=None, defaults=(False,), kwonlyargs=[], kwonlydefaults=None, annotations={'return': typing.Union[float, paddle.Tensor], 'mel': typing.Union[float, paddle.Tensor], 'htk': <class 'bool'>}), ('document', 'e93b432d382f98c60d7c7599489e7072'))
paddle.audio.functional.functional.power_to_db (ArgSpec(args=['spect', 'ref_value', 'amin', 'top_db'], varargs=None, varkw=None, defaults=(1.0, 1e-10, 80.0), kwonlyargs=[], kwonlydefaults=None, annotations={'return': <class 'paddle.Tensor'>, 'spect': <class 'paddle.Tensor'>, 'ref_value': <class 'float'>, 'amin': <class 'float'>, 'top_db': typing.Union[float, NoneType]}), ('document', '28bbb1973e8399e856bfaea0415cecb9'))
paddle.audio.functional.window.get_window (ArgSpec(args=['window', 'win_length', 'fftbins', 'dtype'], varargs=None, varkw=None, defaults=(True, 'float64'), kwonlyargs=[], kwonlydefaults=None, annotations={'return': <class 'paddle.Tensor'>, 'window': typing.Union[str, typing.Tuple[str, float]], 'win_length': <class 'int'>, 'fftbins': <class 'bool'>, 'dtype': <class 'str'>}), ('document', '2418d63da10c0cd5da9ecf0a88ddf783'))
