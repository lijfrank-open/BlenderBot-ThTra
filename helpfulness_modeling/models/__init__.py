try:
    from helpfulness_modeling.models.disagreement import Model as dPM
    from helpfulness_modeling.models.soft_label import Model as soft
    from helpfulness_modeling.models.major_voting import Model as major
    from helpfulness_modeling.models.single_training import Model as single
except ModuleNotFoundError:
    from models.disagreement import Model as dPM
    from models.soft_label import Model as soft
    from models.major_voting import Model as major
    from models.single_training import Model as single
models = {
    "d-PM": dPM,
    "soft": soft,
    "major": major,
    "single": single,
}
