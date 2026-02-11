from .get_tierpsy_features import get_tierpsy_features


def args_(fn, param):
    # getWormFeatures
    main_func = get_tierpsy_features
    requirements = ['FEAT_INIT']

    # FOV splitting
    splitfov_json_path = param.p_dict['MWP_mapping']
    if splitfov_json_path == 'custom':  # special case of custom json
        splitfov_json_path = param.p_dict['MWP_path_to_custom_mapping']

    #arguments used by AnalysisPoints.py
    return {
          'func': main_func,
          'argkws': {
                    'features_file': fn['featuresN'],
                    'derivate_delta_time': param.p_dict['feat_derivate_delta_time'],
                    'splitfov_json_path': splitfov_json_path
                    },
          'input_files' : [fn['featuresN']],
          'output_files': [fn['featuresN']],
          'requirements' : requirements
      }
