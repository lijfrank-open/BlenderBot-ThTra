try:
    from helpfulness_modeling.inputters.esc import Inputter as esc
    from helpfulness_modeling.inputters.mic import Inputter as mic
    from helpfulness_modeling.inputters.single import Inputter as single
except ModuleNotFoundError:
    from inputters.esc import Inputter as esc
    from inputters.mic import Inputter as mic
    from inputters.single import Inputter as single
inputters = {
    "esc": esc,
    "mic": mic,
    "single": single,
}
