import ml_collections


def get_contrast_transform_config():
    config = {
        'img_size': 512,  # first resize according to this
        'rotation': (-180, 180),
        'flip': True,
        'heavy_data_augmentation': {
            'color_jitter': {
                'brightness': 0.4,  # how much to jitter brightness
                'contrast': 0.4,  # How much to jitter contrast
                'saturation': 0.4,
                'hue': 0.1,
            },
            'resized_crop': {
                'input_size': 224,  # size of RandomResizedCrop, will affect final resolution
                'scale': (0.3, 1.2),  # range of size of the origin size cropped
                'ratio': (0.3, 1.2),  # range of aspect ratio of the origin aspect ratio cropped
            },
            # 'affine': {
            #     'degrees': (-180, 180),  # range of degrees to select from
            #     'translate': (0.2, 0.2)  # tuple of maximum absolute fraction for horizontal and vertical translations
            #     },
            'gray': 0.2,
        },
        'mean': [0.425753653049469, 0.29737451672554016, 0.21293757855892181],
        'std': [0.27670302987098694, 0.20240527391433716, 0.1686241775751114],
    }
    return ml_collections.ConfigDict(config)


def get_linear_train_transform_config():
    # abandon translate and gaussion_blur (different from miccai)
    config = {
        'img_size': 512,  # first resize according to this
        'rotation': (-180, 180),
        'flip': True,
        'heavy_data_augmentation': {
            'color_jitter': {
                'brightness': 0.2,  # how much to jitter brightness
                'contrast': 0.2,  # How much to jitter contrast
                'saturation': 0,
                'hue': 0,
            },
            # 'resized_crop': {
            #     'input_size': 512,  # size of RandomResizedCrop
            #     'scale': (0.8, 1.2),  # range of size of the origin size cropped
            #     'ratio': (0.8, 1.2),  # range of aspect ratio of the origin aspect ratio cropped
            # },
            # 'affine': {
            #     'degrees': (-180, 180),  # range of degrees to select from
            #     'translate': (0.2, 0.2)  # tuple of maximum absolute fraction for horizontal and vertical translations
            #     },
            'gray': 0.5,
        },
        'mean': [0.425753653049469, 0.29737451672554016, 0.21293757855892181],
        'std': [0.27670302987098694, 0.20240527391433716, 0.1686241775751114],
    }
    return ml_collections.ConfigDict(config)


def get_linear_test_transform_config():
    config = {
        'img_size': 512,  # first resize according to this
        # 'rotation': (-180, 180),
        # 'flip': True,
        # 'heavy_data_augmentation': {
        #     'color_jitter': {
        #         'brightness': 0.4,  # how much to jitter brightness
        #         'contrast': 0.4,  # How much to jitter contrast
        #         'saturation': 0.4,
        #         'hue': 0.1,
        #     },
        #     'resized_crop': {
        #         'input_size': 512,  # size of RandomResizedCrop
        #         'scale': (0.8, 1.2),  # range of size of the origin size cropped
        #         'ratio': (0.8, 1.2),  # range of aspect ratio of the origin aspect ratio cropped
        #     },
        #     # 'affine': {
        #     #     'degrees': (-180, 180),  # range of degrees to select from
        #     #     'translate': (0.2, 0.2)  # tuple of maximum absolute fraction for horizontal and vertical translations
        #     #     },
        #     'gray': 0.2,
        # },
        'mean': [0.425753653049469, 0.29737451672554016, 0.21293757855892181],
        'std': [0.27670302987098694, 0.20240527391433716, 0.1686241775751114],
    }
    return ml_collections.ConfigDict(config)